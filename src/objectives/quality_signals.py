"""
QuoTe Quality Signals Module

Computes the two quality signals used in Step C of the QuoTe pipeline:

    ZOL_S(E) = h(E)                          (zero-order: direct risk score)
    FOL_S(E) = ε · ‖∇_E h(E)‖₂              (first-order: gradient sensitivity)

Interpretation:
- High ZOL → model already at high compliance risk on this input.
- High FOL → small perturbation causes large risk change → near decision boundary.
- Low FOL  → model is confident (either safely refusing or clearly complying).

Design constraints:
- Gradient computed with respect to perturbed_embeds (leaf tensor, requires_grad=True).
- Model weights remain frozen throughout.
- FOL computed over the *full* embedding (all token positions) then masked to
  perturbed positions only, matching the L2 projection convention.
"""

import logging
from dataclasses import dataclass

import torch

from embedding.embedding_perturb import EmbedBatch
from materialization.model_loader import LoadedModel
from objectives.safety_risk import compute_h_E_for_embeds

logger = logging.getLogger(__name__)


# =============================================================================
# Result types
# =============================================================================


@dataclass
class QualitySignals:
    """
    Quality signal values for a batch of behaviors.

    Attributes:
        zol: (N,) float32 — zero-order loss = h(E) per sample.
        fol: (N,) float32 — first-order loss = ε · ‖∇_E h(E)‖₂ per sample.
        h_mean: Scalar float — batch mean h(E).
    """

    zol: torch.Tensor  # (N,) zero-order per-sample
    fol: torch.Tensor  # (N,) first-order per-sample
    h_mean: float


# =============================================================================
# Core computation
# =============================================================================


def compute_quality_signals(
    embed_batch: EmbedBatch,
    loaded: LoadedModel,
    refusal_token_ids: torch.Tensor,
    epsilon: float,
    grad_clip: float = 1.0,
) -> QualitySignals:
    """
    Compute ZOL and FOL quality signals for a batch of embedded behaviors.

    Steps:
      1. Forward pass through frozen model using perturbed_embeds.
      2. Compute h(E) and per-sample h values.
      3. Backpropagate to obtain ∇_E h(E) w.r.t. perturbed_embeds.
      4. Apply perturbation mask to gradient (only perturb-eligible positions).
      5. Clip gradient globally.
      6. ZOL_S = h_per_sample (detached).
      7. FOL_S = ε · ‖masked_grad‖₂ per sample.

    Args:
        embed_batch: EmbedBatch from embedding_perturb.embed_behaviors().
            perturbed_embeds must have requires_grad=True.
        loaded: Frozen LoadedModel.
        refusal_token_ids: 1-D tensor of refusal token IDs.
        epsilon: L2 perturbation ball radius (ε).
        grad_clip: Global L2 gradient clipping norm.

    Returns:
        QualitySignals with zol, fol, and h_mean.
    """
    assert embed_batch.perturbed_embeds.requires_grad, (
        "perturbed_embeds must have requires_grad=True for FOL computation."
    )

    model = loaded.model
    device = loaded.device

    # Determine response evaluation position (per-sample)
    # For simplicity use the minimum response position across the batch
    response_pos = int(embed_batch.response_positions.min().item())

    # ------------------------------------------------------------------
    # Step 1 & 2: Forward pass + h(E)
    # ------------------------------------------------------------------
    h_E, h_per_sample = compute_h_E_for_embeds(
        model=model,
        perturbed_embeds=embed_batch.perturbed_embeds,
        attention_mask=embed_batch.attention_mask,
        refusal_token_ids=refusal_token_ids,
        response_position=response_pos,
    )

    # ------------------------------------------------------------------
    # Step 3: Backpropagation
    # ------------------------------------------------------------------
    # Clear any existing gradients on the leaf tensor
    if embed_batch.perturbed_embeds.grad is not None:
        embed_batch.perturbed_embeds.grad.zero_()

    h_E.backward()

    grad = embed_batch.perturbed_embeds.grad  # (B, T, d)
    if grad is None:
        logger.warning(
            "Gradient is None after backward — returning zero FOL. "
            "Ensure perturbed_embeds.requires_grad=True."
        )
        batch_size = embed_batch.perturbed_embeds.shape[0]
        return QualitySignals(
            zol=h_per_sample.cpu(),
            fol=torch.zeros(batch_size, dtype=torch.float32),
            h_mean=float(h_E.detach().item()),
        )

    # ------------------------------------------------------------------
    # Step 4: Apply perturbation mask to gradient
    # Only perturb-eligible positions contribute to the FOL norm.
    # ------------------------------------------------------------------
    mask_expanded = embed_batch.perturbation_mask.to(grad.dtype).unsqueeze(-1)  # (B, T, 1)
    masked_grad = grad * mask_expanded  # (B, T, d)

    # ------------------------------------------------------------------
    # Step 5: Global gradient clipping
    # ------------------------------------------------------------------
    flat = masked_grad.view(masked_grad.shape[0], -1)  # (B, T*d)
    norms = flat.norm(p=2, dim=-1, keepdim=True)  # (B, 1)
    clip_factor = torch.clamp(grad_clip / (norms + 1e-8), max=1.0)
    flat = flat * clip_factor
    masked_grad_clipped = flat.view_as(masked_grad)

    # ------------------------------------------------------------------
    # Step 6 & 7: ZOL and FOL
    # ------------------------------------------------------------------
    # ZOL_S(E) = h(E) per sample
    zol = h_per_sample.cpu()  # (B,) — already detached

    # FOL_S(E) = ε · ‖∇_E h(E)‖₂  (over masked, clipped gradient)
    flat_clipped = masked_grad_clipped.view(masked_grad_clipped.shape[0], -1)
    grad_norms = flat_clipped.norm(p=2, dim=-1)  # (B,)
    fol = (epsilon * grad_norms).detach().float().cpu()  # (B,)

    logger.debug(
        "Quality signals — ZOL: mean=%.4f, FOL: mean=%.4f",
        zol.mean().item(),
        fol.mean().item(),
    )

    return QualitySignals(
        zol=zol,
        fol=fol,
        h_mean=float(h_E.detach().item()),
    )


def compute_quality_signals_no_grad(
    embed_batch: EmbedBatch,
    loaded: LoadedModel,
    refusal_token_ids: torch.Tensor,
    epsilon: float,
    grad_clip: float = 1.0,
) -> QualitySignals:
    """
    Convenience wrapper: temporarily enables gradients on perturbed_embeds,
    computes quality signals, then detaches.

    Use this when embed_batch.perturbed_embeds does not yet have requires_grad.

    Args:
        embed_batch: EmbedBatch (perturbed_embeds may be detached).
        loaded: Frozen LoadedModel.
        refusal_token_ids: 1-D refusal token ID tensor.
        epsilon: L2 ball radius.
        grad_clip: Gradient clipping norm.

    Returns:
        QualitySignals.
    """
    # Temporarily create a grad-enabled leaf from current perturbed_embeds
    tmp_embeds = embed_batch.perturbed_embeds.detach().clone().requires_grad_(True)
    tmp_batch = EmbedBatch(
        input_ids=embed_batch.input_ids,
        original_embeds=embed_batch.original_embeds,
        perturbed_embeds=tmp_embeds,
        attention_mask=embed_batch.attention_mask,
        perturbation_mask=embed_batch.perturbation_mask,
        response_positions=embed_batch.response_positions,
    )
    return compute_quality_signals(
        embed_batch=tmp_batch,
        loaded=loaded,
        refusal_token_ids=refusal_token_ids,
        epsilon=epsilon,
        grad_clip=grad_clip,
    )
