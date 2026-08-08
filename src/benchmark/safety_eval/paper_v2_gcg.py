"""Hard-token state helpers for the dedicated paper-v2 GCG runner."""

from __future__ import annotations

import torch

from .objective import EditableState
from .projection_vocabulary import (
    _tokenizer_special_ids,
    ascii_printable_token_id_allowed,
)
from .prompt_contract import TokenizedEditablePrompt, scatter_editable


def standard_gcg_forbidden_token_ids(tokenizer: object, *, vocabulary_size: int) -> tuple[int, ...]:
    """Return special, non-ASCII, and non-printable IDs excluded by standard GCG."""
    if vocabulary_size < 1:
        raise ValueError("GCG vocabulary must be non-empty")
    special_ids = _tokenizer_special_ids(tokenizer, vocabulary_size)
    return tuple(
        token_id
        for token_id in range(vocabulary_size)
        if not ascii_printable_token_id_allowed(
            tokenizer,
            token_id,
            special_ids=special_ids,
        )
    )


def _validate_hard_ids(token_ids: object, *, name: str) -> torch.Tensor:
    if not isinstance(token_ids, torch.Tensor):
        raise ValueError(f"{name} must be a tensor of integer token IDs")
    if token_ids.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, tokens]")
    if token_ids.dtype == torch.bool or token_ids.is_floating_point() or token_ids.is_complex():
        raise ValueError(f"{name} must contain integer token IDs")
    if token_ids.numel() == 0:
        raise ValueError(f"{name} must contain at least one token ID")
    return token_ids


def build_gcg_initial_state(
    prompt: TokenizedEditablePrompt,
    *,
    prefix_ids: torch.Tensor,
) -> EditableState:
    """Build the discrete GCG state from readable ``z`` and annotated ``U`` IDs."""
    z = _validate_hard_ids(prefix_ids, name="prefix IDs")
    u = _validate_hard_ids(prompt.gather_editable_ids(), name="annotated U IDs").to(z.device)
    if z.shape[0] != u.shape[0]:
        raise ValueError("GCG z/U IDs must share a batch dimension")
    if u.shape[1] != len(prompt.editable_positions):
        raise ValueError("GCG U IDs must match Omega_s exactly")
    return EditableState(
        z=z,
        u=u,
        z0=z.detach().clone(),
        u0=u.detach().clone(),
    )


def reconstruct_gcg_token_ids(
    prompt: TokenizedEditablePrompt,
    *,
    z_token_ids: torch.Tensor,
    u_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return hard IDs in paper order: ``[z; Phi_tilde(p; U)]``."""
    z = _validate_hard_ids(z_token_ids, name="z token IDs")
    u = _validate_hard_ids(u_token_ids, name="U token IDs")
    base = _validate_hard_ids(prompt.base_token_ids, name="base token IDs").to(u.device)
    if z.shape[0] != u.shape[0] or u.shape[0] != base.shape[0]:
        raise ValueError("GCG z, U, and base IDs must share a batch dimension")
    if u.shape[1] != len(prompt.editable_positions):
        raise ValueError("GCG U IDs must match Omega_s exactly")

    reconstructed = scatter_editable(u, base, prompt.editable_positions)
    frozen = torch.tensor(prompt.frozen_positions, dtype=torch.long, device=base.device)
    if not torch.equal(reconstructed.index_select(1, frozen), base.index_select(1, frozen)):
        raise RuntimeError("GCG reconstruction changed frozen token IDs")
    return torch.cat((z.to(reconstructed.device), reconstructed), dim=1)


def audit_gcg_changes(
    initial_z: torch.Tensor,
    initial_u: torch.Tensor,
    current_z: torch.Tensor,
    current_u: torch.Tensor,
) -> dict[str, int | list[int]]:
    """Count changed token positions for an unambiguous single-sample state."""
    initial_z = _validate_hard_ids(initial_z, name="initial z")
    initial_u = _validate_hard_ids(initial_u, name="initial U")
    current_z = _validate_hard_ids(current_z, name="current z")
    current_u = _validate_hard_ids(current_u, name="current U")
    if initial_z.shape != current_z.shape or initial_u.shape != current_u.shape:
        raise ValueError("initial and current GCG IDs must have matching shapes")
    if any(ids.shape[0] != 1 for ids in (initial_z, initial_u, current_z, current_u)):
        raise ValueError("GCG change audits support batch size 1 only")

    z_positions = torch.nonzero(initial_z[0] != current_z[0], as_tuple=False).flatten().tolist()
    u_positions = torch.nonzero(initial_u[0] != current_u[0], as_tuple=False).flatten().tolist()
    return {
        "z": len(z_positions),
        "u": len(u_positions),
        "total": len(z_positions) + len(u_positions),
        "z_positions": z_positions,
        "u_positions": u_positions,
    }
