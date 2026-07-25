"""
Soft-Prefix + Editable Seed Block Joint Optimisation

Adam-based gradient optimisation of the continuous attack state u = [z; U].

Loss (to maximise, so we negate for minimisation):
    L(u) = r̃(u) − γ_z · ‖z‖² − γ_u · ‖U − U₀‖²

where r̃(u) is the dual-anchor proxy risk:
    r̃(u) = avg_log_P(Y_ans | H(u)) − avg_log_P(Y_ref | H(u))
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass

import torch

from embedding.attack_state import AttackState
from config.quote_config import QuoTeConfig
from materialization.model_loader import LoadedModel
from objectives.proxy_risk import compute_fol_from_embeds
from objectives.safety_risk import (
    build_anchor_token_ids,
    compute_dual_anchor_risk,
)
from embedding.soft_prefix import build_full_input

logger = logging.getLogger(__name__)


@dataclass
class OptStep:
    """Snapshot of one optimisation step (for trace logging)."""
    step: int
    proxy_risk: float
    fol: float
    prefix_norm: float
    seed_drift: float
    loss: float


def optimise_state(
    state: AttackState,
    loaded: LoadedModel,
    answer_token_ids: torch.Tensor,
    refusal_token_ids: torch.Tensor,
    config: QuoTeConfig,
    wildguard_judge: object | None = None,
) -> tuple[AttackState, list[OptStep]]:
    """Run Adam optimisation on u = [z; U] in-place.

    Args:
        state: AttackState with soft_prefix (z), editable_seed_block (U),
               initial_seed_block (U₀), and frozen_scaffold_ids set.
        loaded: Frozen surrogate model.
        answer_token_ids: Y_ans token IDs.
        refusal_token_ids: Y_ref token IDs.
        config: Hyperparameters.
        wildguard_judge: Optional WildGuardJudge for periodic calibration.

    Returns:
        (updated_state, trace)
    """
    assert state.soft_prefix is not None, "soft_prefix must be initialised"
    assert state.editable_seed_block is not None, "editable_seed_block must be initialised"
    assert state.initial_seed_block is not None, "initial_seed_block must be set"
    assert state.frozen_scaffold_ids is not None, "frozen_scaffold_ids must be set"

    z = state.soft_prefix.detach().clone().requires_grad_(True)
    U = state.editable_seed_block.detach().clone().requires_grad_(True)
    U0 = state.initial_seed_block.detach()  # frozen reference

    optimizer = torch.optim.Adam([z, U], lr=config.lr)
    trace: list[OptStep] = []

    best_loss = float("-inf")
    patience_counter = 0
    _t_start = _time.monotonic()

    # Pre-cache frozen scaffold embeddings (unchanged across all steps)
    with torch.no_grad():
        _cached_scaffold_embeds = loaded.embed_layer(
            state.frozen_scaffold_ids
        ).to(z.dtype)

    for step in range(config.max_opt_steps):
        optimizer.zero_grad()

        # Build H(u) = [z; cached_scaffold_embeds; U] — scaffold is pre-cached
        full_embeds = torch.cat([z, _cached_scaffold_embeds, U], dim=1)
        total_len = full_embeds.shape[1]
        attn_mask = torch.ones(1, total_len, dtype=torch.long, device=loaded.device)

        # Dual-anchor proxy risk r̃(u)
        r_tilde = compute_dual_anchor_risk(
            model=loaded.model,
            perturbed_embeds=full_embeds,
            attention_mask=attn_mask,
            answer_token_ids=answer_token_ids,
            refusal_token_ids=refusal_token_ids,
        )

        proxy_val = r_tilde.item()

        # Regularisation terms
        reg_z = config.gamma_z * z.norm(p=2) ** 2
        reg_u = config.gamma_u * (U - U0).norm(p=2) ** 2

        # Branch-aware FOL term in the loss objective (second-order gradient)
        # HV branch: O⁻ = r̃ − λ·FOL − reg  (penalises uncertainty → settled maxima)
        # BD branch: O⁺ = r̃ + λ·FOL − reg  (rewards uncertainty → decision boundary)
        _is_log_step = (step % 10 == 0) or (step == config.max_opt_steps - 1)
        if config.lambda_fol > 0.0 and state.branch_type in ("high_value", "boundary"):
            fol_grads = torch.autograd.grad(
                r_tilde, [z, U], create_graph=True, retain_graph=True
            )
            fol_tensor = config.epsilon * (
                fol_grads[0].norm(p=2) ** 2 + fol_grads[1].norm(p=2) ** 2
            ).sqrt()
            fol_val = fol_tensor.item()
            sign = -1.0 if state.branch_type == "high_value" else +1.0
            loss = -(r_tilde + sign * config.lambda_fol * fol_tensor - reg_z - reg_u)
        else:
            fol_val = 0.0
            loss = -(r_tilde - reg_z - reg_u)

        loss.backward()

        p_norm = z.detach().norm(p=2).item()
        s_drift = (U.detach() - U0).norm(p=2).pow(2).item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([z, U], config.grad_clip)
        optimizer.step()

        step_log = OptStep(
            step=step,
            proxy_risk=proxy_val,
            fol=fol_val,
            prefix_norm=p_norm,
            seed_drift=s_drift,
            loss=-loss.item(),
        )
        trace.append(step_log)

        if _is_log_step:
            _elapsed = _time.monotonic() - _t_start
            logger.info(
                "  opt step %d/%d  r̃=%.4f fol=%.4f |z|=%.3f drift=%.3f  [%.1fs]",
                step, config.max_opt_steps, proxy_val, fol_val, p_norm, s_drift, _elapsed,
            )

        # WildGuard calibration every judge_interval steps
        if wildguard_judge is not None and (step + 1) % config.judge_interval == 0:
            _do_wildguard_calibration(state, z, U, loaded, wildguard_judge, step)

        # Early stopping
        if -loss.item() > best_loss + 1e-4:
            best_loss = -loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.early_stop_patience:
            logger.info("Early stop at step %d (patience=%d)", step, config.early_stop_patience)
            break

    # Update state with final values
    state.soft_prefix = z.detach().clone()
    state.editable_seed_block = U.detach().clone()
    state.proxy_risk = trace[-1].proxy_risk if trace else -1.0
    state.fol = trace[-1].fol if trace else -1.0
    state.zol = state.proxy_risk
    state.prefix_norm = z.detach().norm(p=2).item()
    state.seed_block_drift = (U.detach() - U0).norm(p=2).pow(2).item()
    state.step = len(trace)

    return state, trace


def _do_wildguard_calibration(
    state: AttackState,
    z: torch.Tensor,
    U: torch.Tensor,
    loaded: LoadedModel,
    wildguard_judge: object,
    step: int,
) -> None:
    """Decode a few tokens from the surrogate and check with WildGuard."""
    from soft_prefix import build_full_input

    with torch.no_grad():
        full_embeds, attn_mask = build_full_input(
            soft_prefix=z.detach(),
            frozen_scaffold_ids=state.frozen_scaffold_ids,
            editable_seed_block=U.detach(),
            loaded=loaded,
        )
        # Greedy decode a short response
        outputs = loaded.model.generate(
            inputs_embeds=full_embeds,
            attention_mask=attn_mask,
            max_new_tokens=64,
            do_sample=False,
        )
        # Decode generated tokens (skip input length)
        gen_ids = outputs[0, full_embeds.shape[1]:]
        response_text = loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Score with WildGuard
    if hasattr(wildguard_judge, "is_unsafe"):
        is_unsafe = wildguard_judge.is_unsafe(state.original_seed, response_text)
        logger.info(
            "  WildGuard @step %d: unsafe=%s response='%.80s…'",
            step, is_unsafe, response_text,
        )
