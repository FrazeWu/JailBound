"""Lossless tensor-to-token projection primitives for safety evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from .objective import EditableState
from .prompt_contract import TokenizedEditablePrompt
from .schema import FailureKind, MaterializationRecord, RecordStatus, TransportType, V2MaterializationRecord


@dataclass(frozen=True)
class ContinuousMaterialization:
    """Projection evidence for both continuous editable blocks."""

    prefix_token_ids: tuple[int, ...]
    seed_token_ids: tuple[int, ...]
    prefix_projection_cosine: float
    seed_projection_cosine: float


@dataclass(frozen=True)
class DiscreteCandidate:
    """Already-discrete prefix and seed token identities."""

    prefix_token_ids: tuple[int, ...]
    seed_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ContinuousCandidate:
    """Continuous editable blocks and the vocabulary used to project them."""

    state: EditableState
    vocabulary_embeddings: torch.Tensor
    forbidden_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class V2CandidateMaterialization:
    """One-pass reconstruction evidence for an annotated prompt contract."""

    projected_z_token_ids: tuple[int, ...]
    projected_u_token_ids: tuple[int, ...]
    reconstructed_base_token_ids: tuple[int, ...]
    complete_token_ids: tuple[int, ...]
    flat_prompt: str
    frozen_positions_unchanged: bool


def meets_semantic_threshold(similarity: float, *, threshold: float) -> bool:
    """Return whether a semantic similarity meets its frozen acceptance threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("semantic threshold must be in [0, 1]")
    return float(similarity) >= threshold


def materialize_v2_candidate(
    *,
    candidate: ContinuousCandidate | DiscreteCandidate,
    prompt: TokenizedEditablePrompt,
    tokenizer: Any,
    special_token_ids: Iterable[int] = (),
) -> V2CandidateMaterialization:
    """Project both editable blocks and reconstruct ``[z; Phi(p; U)]`` once."""
    if isinstance(candidate, ContinuousCandidate):
        evidence = materialize_continuous_state(
            candidate.state,
            candidate.vocabulary_embeddings,
            forbidden_token_ids=tuple(special_token_ids) + candidate.forbidden_token_ids,
        )
        z_ids, u_ids = evidence.prefix_token_ids, evidence.seed_token_ids
    else:
        z_ids = tuple(int(token_id) for token_id in candidate.prefix_token_ids)
        u_ids = tuple(int(token_id) for token_id in candidate.seed_token_ids)
    if len(u_ids) != len(prompt.editable_positions):
        raise ValueError("projected U token count must match editable prompt positions")
    forbidden = {int(token_id) for token_id in special_token_ids}
    if forbidden.intersection(z_ids + u_ids):
        raise ValueError("projected editable tokens must not include forbidden IDs")
    base = tuple(int(token_id) for token_id in prompt.base_token_ids[0].detach().cpu().tolist())
    reconstructed = list(base)
    for position, token_id in zip(prompt.editable_positions, u_ids):
        reconstructed[position] = token_id
    frozen_unchanged = all(reconstructed[position] == base[position] for position in prompt.frozen_positions)
    if not frozen_unchanged:
        raise AssertionError("v2 materialization changed a frozen prompt token")
    complete = z_ids + tuple(reconstructed)
    decoded = str(tokenizer.decode(list(complete), skip_special_tokens=True)).strip()
    if not decoded:
        raise ValueError("decoded v2 candidate must be non-empty")
    return V2CandidateMaterialization(
        projected_z_token_ids=z_ids,
        projected_u_token_ids=u_ids,
        reconstructed_base_token_ids=tuple(reconstructed),
        complete_token_ids=complete,
        flat_prompt=decoded,
        frozen_positions_unchanged=True,
    )


def build_v2_materialization_record(
    *,
    result: V2CandidateMaterialization,
    prompt: TokenizedEditablePrompt,
    run_id: str,
    config_hash: str,
    sample_id: str,
    source: str,
    method: str,
    branch: str,
    step: int,
    transport: TransportType = TransportType.text,
    full_prompt_similarity: float = 1.0,
    editable_span_similarity: float = 1.0,
) -> V2MaterializationRecord:
    """Turn one v2 reconstruction into a strict materialization ledger row."""
    return V2MaterializationRecord(
        schema_version="reviewer_eval.v2",
        run_id=run_id,
        config_hash=config_hash,
        sample_id=sample_id,
        source=source,
        method=method,
        branch=branch,
        step=step,
        transport=transport,
        editable_positions=prompt.editable_positions,
        original_token_ids=tuple(int(value) for value in prompt.base_token_ids[0].tolist()),
        projected_z_token_ids=result.projected_z_token_ids,
        projected_u_token_ids=result.projected_u_token_ids,
        reconstructed_base_token_ids=result.reconstructed_base_token_ids,
        complete_token_ids=result.complete_token_ids,
        frozen_positions_unchanged=result.frozen_positions_unchanged,
        span_boundary_expansions=prompt.boundary_expansions,
        full_prompt_similarity=full_prompt_similarity,
        editable_span_similarity=editable_span_similarity,
        flat_prompt=result.flat_prompt,
        status=RecordStatus.complete,
        failure_kind=None,
        failure_reason=None,
    )


def build_materialization_record(
    *,
    schema_version: str,
    run_id: str,
    config_hash: str,
    sample_id: str,
    source: str,
    method: str,
    checkpoint: int,
    system_prompt: str,
    user_prompt: str,
    flat_prompt: str,
    semantic_similarity_before: float,
    semantic_similarity_after: float,
    semantic_threshold: float,
    category_before: str,
    category_after: str,
    candidate: ContinuousCandidate | DiscreteCandidate,
    special_token_ids: Iterable[int] = (),
    projection_attack_score_before: float | None = None,
    projection_attack_score_after: float | None = None,
) -> MaterializationRecord:
    """Build a strict record for an already-discrete candidate."""
    accepted = meets_semantic_threshold(semantic_similarity_after, threshold=semantic_threshold)
    if isinstance(candidate, ContinuousCandidate):
        evidence = materialize_continuous_state(
            candidate.state,
            candidate.vocabulary_embeddings,
            forbidden_token_ids=candidate.forbidden_token_ids,
        )
        prefix_token_ids = evidence.prefix_token_ids
        seed_token_ids = evidence.seed_token_ids
        prefix_projection_cosine: float | None = evidence.prefix_projection_cosine
        seed_projection_cosine: float | None = evidence.seed_projection_cosine
    else:
        prefix_token_ids = tuple(int(token_id) for token_id in candidate.prefix_token_ids)
        seed_token_ids = tuple(int(token_id) for token_id in candidate.seed_token_ids)
        prefix_projection_cosine = 1.0 if prefix_token_ids else None
        seed_projection_cosine = 1.0 if seed_token_ids else None
    candidate_token_ids = prefix_token_ids + seed_token_ids
    special_ids = {int(token_id) for token_id in special_token_ids}

    failure_kind: FailureKind | None = None
    failure_reason: str | None = None
    if not flat_prompt.strip() or not candidate_token_ids:
        failure_kind = FailureKind.materialization
        failure_reason = "empty candidate"
    elif all(token_id in special_ids for token_id in candidate_token_ids):
        failure_kind = FailureKind.materialization
        failure_reason = "special-only candidate"
    elif not accepted:
        failure_kind = FailureKind.semantic_filter
        failure_reason = "below semantic threshold"

    return MaterializationRecord(
        schema_version=schema_version,
        run_id=run_id,
        config_hash=config_hash,
        sample_id=sample_id,
        source=source,
        method=method,
        checkpoint=checkpoint,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        flat_prompt=flat_prompt,
        prefix_token_ids=prefix_token_ids,
        seed_token_ids=seed_token_ids,
        prefix_projection_cosine=prefix_projection_cosine,
        seed_projection_cosine=seed_projection_cosine,
        semantic_similarity_before=semantic_similarity_before,
        semantic_similarity_after=semantic_similarity_after,
        category_before=category_before,
        category_after=category_after,
        intent_preserved=failure_kind is None,
        projection_attack_score_before=projection_attack_score_before,
        projection_attack_score_after=projection_attack_score_after,
        status=RecordStatus.failed if failure_kind is not None else RecordStatus.complete,
        failure_kind=failure_kind,
        failure_reason=failure_reason,
    )


def _state_from_payload(payload: Mapping[str, object]) -> EditableState:
    try:
        z, u = payload["z"], payload["u"]
    except KeyError as error:
        raise ValueError("checkpoint state is missing editable tensors") from error
    if not isinstance(z, torch.Tensor) or not isinstance(u, torch.Tensor):
        raise ValueError("checkpoint editable states must be tensors")
    if z.ndim != 3 or u.ndim != 3 or z.shape[0] != u.shape[0] or z.shape[-1] != u.shape[-1]:
        raise ValueError("checkpoint editable states have incompatible shapes")
    return EditableState(z=z, u=u, z0=z.detach().clone(), u0=u.detach().clone())


def _discrete_candidate_from_payload(payload: Mapping[str, object]) -> DiscreteCandidate | None:
    z_ids, u_ids = payload.get("z_token_ids"), payload.get("u_token_ids")
    if not isinstance(z_ids, torch.Tensor) or not isinstance(u_ids, torch.Tensor):
        return None
    if z_ids.numel() == 0 or u_ids.numel() == 0:
        return None
    if z_ids.ndim != 2 or u_ids.ndim != 2:
        raise ValueError("checkpoint token IDs must have shape [batch, tokens]")
    return DiscreteCandidate(
        prefix_token_ids=tuple(int(token_id) for token_id in z_ids.detach().reshape(-1).cpu().tolist()),
        seed_token_ids=tuple(int(token_id) for token_id in u_ids.detach().reshape(-1).cpu().tolist()),
    )


def materialize_checkpoint(
    *,
    state_payload: Mapping[str, object],
    vocabulary_embeddings: torch.Tensor,
    tokenizer: Any,
    schema_version: str,
    run_id: str,
    config_hash: str,
    sample_id: str,
    source: str,
    method: str,
    checkpoint: int,
    original_prompt: str,
    category: str,
    semantic_similarity: float,
    semantic_threshold: float,
) -> MaterializationRecord:
    """Materialize one saved optimizer state without exposing text at the call site.

    Discrete baselines retain their selected IDs. Continuous methods are
    projected once against the local vocabulary, excluding tokenizer special
    tokens. The caller supplies the already-computed semantic score so encoder
    choice and calibration remain explicit in the stage orchestration.
    """
    if not isinstance(original_prompt, str) or not original_prompt.strip():
        raise ValueError("original prompt must be a non-empty string")
    special_ids = tuple(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ()))
    candidate = _discrete_candidate_from_payload(state_payload)
    if candidate is None:
        state = _state_from_payload(state_payload)
        if state.z.device != vocabulary_embeddings.device or state.u.device != vocabulary_embeddings.device:
            state = EditableState(
                z=state.z.to(vocabulary_embeddings.device),
                u=state.u.to(vocabulary_embeddings.device),
                z0=state.z0.to(vocabulary_embeddings.device),
                u0=state.u0.to(vocabulary_embeddings.device),
            )
        candidate = ContinuousCandidate(
            state=state,
            vocabulary_embeddings=vocabulary_embeddings,
            forbidden_token_ids=special_ids,
        )
        evidence = materialize_continuous_state(
            candidate.state,
            candidate.vocabulary_embeddings,
            forbidden_token_ids=candidate.forbidden_token_ids,
        )
        prefix_ids, seed_ids = evidence.prefix_token_ids, evidence.seed_token_ids
    else:
        prefix_ids, seed_ids = candidate.prefix_token_ids, candidate.seed_token_ids
    prefix = str(tokenizer.decode(list(prefix_ids), skip_special_tokens=True)).strip()
    seed = str(tokenizer.decode(list(seed_ids), skip_special_tokens=True)).strip()
    flat_prompt = " ".join(part for part in (prefix, original_prompt.strip(), seed) if part)
    return build_materialization_record(
        schema_version=schema_version,
        run_id=run_id,
        config_hash=config_hash,
        sample_id=sample_id,
        source=source,
        method=method,
        checkpoint=checkpoint,
        system_prompt="",
        user_prompt=flat_prompt,
        flat_prompt=flat_prompt,
        semantic_similarity_before=1.0,
        semantic_similarity_after=semantic_similarity,
        semantic_threshold=semantic_threshold,
        category_before=category,
        category_after=category,
        candidate=candidate,
        special_token_ids=special_ids,
    )


def calibrate_threshold(
    positive_similarities: Iterable[float], *, target_recall: float
) -> float:
    """Freeze the lowest similarity retaining the requested positive recall."""
    values = sorted((float(value) for value in positive_similarities), reverse=True)
    if not values:
        raise ValueError("semantic calibration requires accepted positive pairs")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    retained = max(1, int(__import__("math").ceil(len(values) * target_recall)))
    return values[retained - 1]


def _allowed_vocabulary(
    vocabulary_embeddings: torch.Tensor,
    forbidden_token_ids: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if vocabulary_embeddings.ndim != 2:
        raise ValueError("vocabulary embeddings must have shape [vocabulary, hidden]")
    forbidden = set(int(token_id) for token_id in forbidden_token_ids)
    ids = torch.tensor(
        [index for index in range(vocabulary_embeddings.shape[0]) if index not in forbidden],
        device=vocabulary_embeddings.device,
        dtype=torch.long,
    )
    if not len(ids):
        raise ValueError("no allowed vocabulary ids remain after masking")
    return ids, vocabulary_embeddings.index_select(0, ids)


def _project_block(
    block: torch.Tensor,
    allowed_ids: torch.Tensor,
    allowed_embeddings: torch.Tensor,
) -> tuple[tuple[int, ...], float]:
    positions = block.detach().reshape(-1, block.shape[-1]).float()
    vocabulary = allowed_embeddings.detach().float()
    positions = torch.nn.functional.normalize(positions, dim=-1)
    vocabulary = torch.nn.functional.normalize(vocabulary, dim=-1)
    similarities = positions @ vocabulary.T
    maxima, local_ids = similarities.max(dim=-1)
    token_ids = allowed_ids.index_select(0, local_ids).detach().cpu().tolist()
    return tuple(int(token_id) for token_id in token_ids), float(maxima.mean().cpu())


def materialize_continuous_state(
    state: EditableState,
    vocabulary_embeddings: torch.Tensor,
    *,
    forbidden_token_ids: Iterable[int] = (),
) -> ContinuousMaterialization:
    """Project *both* ``z`` and ``u`` against one masked vocabulary."""
    allowed_ids, allowed_embeddings = _allowed_vocabulary(
        vocabulary_embeddings, forbidden_token_ids
    )
    prefix_ids, prefix_cosine = _project_block(state.z, allowed_ids, allowed_embeddings)
    seed_ids, seed_cosine = _project_block(state.u, allowed_ids, allowed_embeddings)
    return ContinuousMaterialization(
        prefix_token_ids=prefix_ids,
        seed_token_ids=seed_ids,
        prefix_projection_cosine=prefix_cosine,
        seed_projection_cosine=seed_cosine,
    )
