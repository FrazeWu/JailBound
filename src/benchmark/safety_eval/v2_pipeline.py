"""V2 post-optimization adapters that preserve annotated editable spans."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .materialization import (
    ContinuousCandidate,
    DiscreteCandidate,
    build_v2_materialization_record,
    materialize_v2_candidate,
    vocabulary_embedding_sha256,
)
from .objective import EditableState
from .prompt_contract import tokenize_editable_prompt
from .io import JsonlLedger, canonical_hash, read_jsonl, sha256_file
from .runtime import validate_v2_provenance_ledgers
from .schema import (
    OptimizationRecord,
    RecordStatus,
    V2BenchmarkExample,
    V2MaterializationRecord,
)


_INTEGRAL_TOKEN_ID_DTYPES = frozenset(
    dtype
    for dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        getattr(torch, "uint16", None),
        getattr(torch, "uint32", None),
        getattr(torch, "uint64", None),
    )
    if dtype is not None
)


def materialize_v2_optimization_state(
    optimization: OptimizationRecord,
    *,
    example: V2BenchmarkExample,
    vocabulary_embeddings: torch.Tensor,
    tokenizer: Any,
    surrogate_tokenizer_sha256: str,
) -> V2MaterializationRecord:
    """Project one terminal single-branch state without modifying frozen tokens."""
    if optimization.schema_version != "reviewer_eval.v2":
        raise ValueError("v2 materialization requires a v2 optimization record")
    if (optimization.sample_id, optimization.source) != (
        example.example_id,
        example.source,
    ):
        raise ValueError("optimization record does not match its immutable example")
    if not optimization.state_path:
        raise ValueError("v2 optimization state path is required")
    if optimization.state_sha256 is None:
        raise ValueError("v2 optimization state sha256 is required")
    _validate_vocabulary_embedding_contract(vocabulary_embeddings, tokenizer)
    state_path = Path(optimization.state_path)
    if sha256_file(state_path) != optimization.state_sha256:
        raise ValueError("v2 optimization state sha256 does not match state file")
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("v2 optimization state must be a mapping")
    checkpoint_embedding_sha256 = payload.get("input_embedding_sha256")
    if not isinstance(checkpoint_embedding_sha256, str):
        raise ValueError("v2 optimization state requires input embedding sha256")
    if checkpoint_embedding_sha256 != vocabulary_embedding_sha256(vocabulary_embeddings):
        raise ValueError("v2 checkpoint input embedding sha256 does not match materializer")
    z, u = payload.get("z"), payload.get("u")
    if not isinstance(z, torch.Tensor) or not isinstance(u, torch.Tensor):
        raise ValueError("v2 optimization state requires z and u tensors")
    prompt = tokenize_editable_prompt(
        example.attack_text, example.editable_spans, tokenizer, surrogate_tokenizer_sha256
    )
    _validate_checkpoint_contract(payload, prompt, example)
    forbidden_token_ids = tuple(
        int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())
    )
    hard_candidate = _validate_saved_hard_token_state(
        payload,
        vocabulary_embeddings=vocabulary_embeddings,
        forbidden_token_ids=forbidden_token_ids,
    )
    z = z.to(vocabulary_embeddings.device)
    u = u.to(vocabulary_embeddings.device)
    result = materialize_v2_candidate(
        candidate=hard_candidate
        if hard_candidate is not None
        else ContinuousCandidate(
            state=EditableState(
                z=z, u=u, z0=z.detach().clone(), u0=u.detach().clone()
            ),
            vocabulary_embeddings=vocabulary_embeddings,
            forbidden_token_ids=forbidden_token_ids,
        ),
        prompt=prompt,
        tokenizer=tokenizer,
        special_token_ids=forbidden_token_ids,
    )
    branch = optimization.representation.rsplit(":", 1)[-1]
    return build_v2_materialization_record(
        result=result,
        prompt=prompt,
        run_id=optimization.run_id,
        config_hash=optimization.config_hash,
        sample_id=optimization.sample_id,
        source=optimization.source,
        method=optimization.method,
        branch=branch,
        step=optimization.checkpoint,
        state_sha256=optimization.state_sha256,
        surrogate_tokenizer_sha256=surrogate_tokenizer_sha256,
        surrogate_embedding_sha256=vocabulary_embedding_sha256(vocabulary_embeddings),
    )


def materialize_v2_terminal_records(
    output_root: str | Path,
    *,
    source: str,
    method: str,
    vocabulary_embeddings: torch.Tensor,
    tokenizer: Any,
    surrogate_tokenizer_sha256: str,
) -> tuple[V2MaterializationRecord, ...]:
    """Materialize each single-branch terminal v2 state once, resumably."""
    root = Path(output_root)
    validate_v2_provenance_ledgers(root)
    examples = {
        example.example_id: example
        for example in (
            V2BenchmarkExample.model_validate(row)
            for row in read_jsonl(root / "manifests" / "v2" / f"controlled_{source}.jsonl")
        )
    }
    expected_step = 0 if method == "init" else 100
    records = [
        OptimizationRecord.model_validate(row)
        for row in read_jsonl(root / "optimization" / source / method / "records.jsonl")
    ]
    terminal = [
        record
        for record in records
        if record.schema_version == "reviewer_eval.v2"
        and record.checkpoint == expected_step
        and record.status is RecordStatus.complete
    ]
    if len(terminal) != len(examples):
        raise ValueError("v2 terminal optimization records are incomplete")
    by_sample = {record.sample_id: record for record in terminal}
    if len(by_sample) != len(terminal) or set(by_sample) != set(examples):
        raise ValueError("v2 terminal optimization identities do not match manifest")
    materializations = tuple(
        materialize_v2_optimization_state(
            by_sample[sample_id],
            example=examples[sample_id],
            vocabulary_embeddings=vocabulary_embeddings,
            tokenizer=tokenizer,
            surrogate_tokenizer_sha256=surrogate_tokenizer_sha256,
        )
        for sample_id in sorted(examples)
    )
    ledger = JsonlLedger(
        root / "optimization" / source / method / "materialization.jsonl",
        key_fields=("sample_id", "step", "branch", "transport"),
    )
    for record in materializations:
        ledger.append_once(record.model_dump(mode="json"))
    return materializations


def _validate_checkpoint_contract(
    payload: Mapping[str, object], prompt: Any, example: V2BenchmarkExample
) -> None:
    base = payload.get("base_token_ids")
    positions = payload.get("editable_positions")
    revision = payload.get("tokenizer_revision")
    span_hashes = payload.get("editable_span_hashes")
    expected_hashes = tuple(canonical_hash(span.model_dump(mode="json")) for span in example.editable_spans)
    if not isinstance(base, torch.Tensor) or not torch.equal(base.to(dtype=torch.long, device="cpu"), prompt.base_token_ids.to(dtype=torch.long, device="cpu")):
        raise ValueError("v2 checkpoint base token contract does not match manifest")
    if not isinstance(positions, torch.Tensor) or tuple(int(value) for value in positions.detach().cpu().tolist()) != prompt.editable_positions:
        raise ValueError("v2 checkpoint editable-position contract does not match manifest")
    if revision != prompt.tokenizer_revision:
        raise ValueError("v2 checkpoint tokenizer revision does not match materializer")
    if not isinstance(span_hashes, tuple) or span_hashes != expected_hashes:
        raise ValueError("v2 checkpoint editable-span contract does not match manifest")


def _validate_vocabulary_embedding_contract(
    vocabulary_embeddings: torch.Tensor, tokenizer: Any
) -> None:
    """Ensure projection rows retain the tokenizer's integer-ID mapping."""
    if vocabulary_embeddings.ndim != 2:
        raise ValueError("vocabulary embeddings must have shape [vocabulary, hidden]")
    rows = int(vocabulary_embeddings.shape[0])
    try:
        vocabulary_size = len(tokenizer)
    except TypeError:
        vocabulary_size = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocabulary = get_vocab() if callable(get_vocab) else None
    if isinstance(vocabulary, Mapping):
        ids = tuple(int(token_id) for token_id in vocabulary.values())
        if tuple(sorted(ids)) != tuple(range(len(ids))) or len(ids) > rows:
            raise ValueError("tokenizer vocabulary IDs must be contiguous and within embedding rows")
        return
    if isinstance(vocabulary_size, int) and rows != vocabulary_size:
        raise ValueError("vocabulary embedding row count does not match tokenizer vocabulary")


def _validate_discrete_projection(
    payload: Mapping[str, object], z_ids: tuple[int, ...], u_ids: tuple[int, ...]
) -> None:
    saved_z, saved_u = payload.get("z_token_ids"), payload.get("u_token_ids")
    if not isinstance(saved_z, torch.Tensor) or not isinstance(saved_u, torch.Tensor):
        return
    if saved_z.numel() == 0 and saved_u.numel() == 0:
        return
    if saved_z.ndim != 2 or saved_u.ndim != 2 or saved_z.shape[0] != 1 or saved_u.shape[0] != 1:
        raise ValueError("v2 checkpoint discrete token IDs must have batch size 1")
    if tuple(int(value) for value in saved_z.detach().cpu().flatten().tolist()) != z_ids or tuple(int(value) for value in saved_u.detach().cpu().flatten().tolist()) != u_ids:
        raise ValueError("v2 re-projection does not match checkpoint discrete token IDs")


def _validate_saved_hard_token_state(
    payload: Mapping[str, object],
    *,
    vocabulary_embeddings: torch.Tensor,
    forbidden_token_ids: tuple[int, ...],
) -> DiscreteCandidate | None:
    """Validate optimizer-selected hard IDs and return their discrete candidate.

    PEZ, GBDA, and GCG snapshots store ``state.z/u`` as lookup results from
    their saved hard token IDs.  The IDs are therefore the authoritative
    discrete result; comparing float32 cosine argmax values after a dtype or
    device reload is not a stable integrity check.
    """
    saved_z, saved_u = payload.get("z_token_ids"), payload.get("u_token_ids")
    if saved_z is None and saved_u is None:
        return None
    if not isinstance(saved_z, torch.Tensor) or not isinstance(saved_u, torch.Tensor):
        raise ValueError("v2 checkpoint discrete token IDs must be tensors")
    if saved_z.numel() == 0 and saved_u.numel() == 0:
        return None
    if saved_z.numel() == 0 or saved_u.numel() == 0:
        raise ValueError("v2 checkpoint discrete token IDs must include both z and u")
    if (
        saved_z.ndim != 2
        or saved_u.ndim != 2
        or saved_z.shape[0] != 1
        or saved_u.shape[0] != 1
        or saved_z.dtype not in _INTEGRAL_TOKEN_ID_DTYPES
        or saved_u.dtype not in _INTEGRAL_TOKEN_ID_DTYPES
    ):
        raise ValueError("v2 checkpoint discrete token IDs must have integer batch size 1")
    if (
        vocabulary_embeddings.ndim != 2
        or not vocabulary_embeddings.is_floating_point()
        or not torch.isfinite(vocabulary_embeddings).all().item()
    ):
        raise ValueError("v2 vocabulary embeddings must be finite floating-point values")
    z_ids = saved_z.detach().to(device=vocabulary_embeddings.device, dtype=torch.long)
    u_ids = saved_u.detach().to(device=vocabulary_embeddings.device, dtype=torch.long)
    all_ids = torch.cat((z_ids.flatten(), u_ids.flatten()))
    if torch.any(all_ids < 0).item() or torch.any(all_ids >= vocabulary_embeddings.shape[0]).item():
        raise ValueError("v2 checkpoint hard-token IDs are outside the vocabulary")
    forbidden = set(forbidden_token_ids)
    if any(int(token_id) in forbidden for token_id in all_ids.detach().cpu().tolist()):
        raise ValueError("v2 checkpoint hard-token IDs include forbidden tokens")

    z, u = payload.get("z"), payload.get("u")
    if not isinstance(z, torch.Tensor) or not isinstance(u, torch.Tensor):
        raise ValueError("v2 checkpoint hard-token state requires z and u tensors")
    expected_shapes = (
        (1, z_ids.shape[1], vocabulary_embeddings.shape[1]),
        (1, u_ids.shape[1], vocabulary_embeddings.shape[1]),
    )
    if z.shape != expected_shapes[0] or u.shape != expected_shapes[1]:
        raise ValueError("v2 checkpoint hard-token state shape does not match token IDs")
    if (
        not z.is_floating_point()
        or not u.is_floating_point()
        or not torch.isfinite(z).all().item()
        or not torch.isfinite(u).all().item()
    ):
        raise ValueError("v2 checkpoint hard-token state must contain finite floating-point values")

    # Compare in the persisted state dtype. A bfloat16 snapshot legitimately
    # contains the bfloat16-rounded lookup of a float32 vocabulary, while a
    # float32 snapshot must match its float32 lookup exactly.
    expected_z = torch.nn.functional.embedding(z_ids, vocabulary_embeddings).to(dtype=z.dtype)
    expected_u = torch.nn.functional.embedding(u_ids, vocabulary_embeddings).to(dtype=u.dtype)
    actual_z = z.to(device=vocabulary_embeddings.device)
    actual_u = u.to(device=vocabulary_embeddings.device)
    if not torch.equal(actual_z, expected_z) or not torch.equal(actual_u, expected_u):
        raise ValueError("v2 checkpoint hard-token embedding state does not match saved token IDs")
    return DiscreteCandidate(
        prefix_token_ids=tuple(int(token_id) for token_id in z_ids.flatten().detach().cpu().tolist()),
        seed_token_ids=tuple(int(token_id) for token_id in u_ids.flatten().detach().cpu().tolist()),
    )
