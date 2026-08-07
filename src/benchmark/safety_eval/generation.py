"""Local target-generation helpers with no runner dependencies."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .schema import (
    FailureKind,
    MaterializationRecord,
    RecordStatus,
    ResponseRecord,
    V2MaterializationRecord,
    V2ResponseRecord,
    token_ids_sha256,
)


@dataclass(frozen=True)
class GenerationResult:
    """One locally generated response and its token accounting."""

    response: str
    input_tokens: int
    generated_tokens: int
    used_system_fallback: bool


def shard_ids(ids: Sequence[str], replicas: int) -> tuple[tuple[str, ...], ...]:
    """Assign each identifier to a deterministic replica shard."""
    if replicas < 1:
        raise ValueError("replicas must be at least one")

    shards: list[list[str]] = [[] for _ in range(replicas)]
    for identifier in ids:
        shard_index = int.from_bytes(
            hashlib.sha256(identifier.encode("utf-8")).digest(), "big"
        ) % replicas
        shards[shard_index].append(identifier)
    return tuple(tuple(shard) for shard in shards)


def _first_sequence(token_ids: Any) -> Any:
    """Return the single generated sequence from list- or tensor-like values."""
    return token_ids[0]


def _is_system_role_rejection(error: Exception) -> bool:
    message = str(error).lower()
    return "system" in message and ("role" in message or "message" in message)


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def move_to_model_input_device(input_ids: Any, model: Any) -> Any:
    """Place chat tokens with the input embedding layer when supported."""
    try:
        embedding = model.get_input_embeddings()
        device = getattr(getattr(embedding, "weight", None), "device", None)
    except (AttributeError, TypeError, ValueError):
        return input_ids
    move = getattr(input_ids, "to", None)
    return move(device) if device is not None and callable(move) else input_ids


def embedding_state_hash(inputs_embeds: torch.Tensor) -> str:
    """Return a stable, content-free fingerprint for a continuous input state."""
    if inputs_embeds.ndim != 3:
        raise ValueError("embedding state must have shape [batch, tokens, hidden]")
    normalized = inputs_embeds.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("ascii"))
    digest.update(normalized.numpy().tobytes())
    return digest.hexdigest()


def generate_from_embeddings(
    model: Any,
    tokenizer: Any,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> GenerationResult:
    """Generate greedily from a continuous embedding state without projection."""
    if inputs_embeds.ndim != 3 or attention_mask.shape != inputs_embeds.shape[:2]:
        raise ValueError("embedding inputs and attention mask have incompatible shapes")
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids = _first_sequence(output_ids)
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return GenerationResult(
        response=response,
        input_tokens=int(inputs_embeds.shape[1]),
        generated_tokens=len(generated_ids),
        used_system_fallback=False,
    )


def generate_from_embedding_batch(
    model: Any,
    tokenizer: Any,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[GenerationResult, ...]:
    """Generate greedily for equal-length continuous states in one batch."""
    if inputs_embeds.ndim != 3 or attention_mask.shape != inputs_embeds.shape[:2]:
        raise ValueError("embedding inputs and attention mask have incompatible shapes")
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if len(output_ids) != inputs_embeds.shape[0]:
        raise ValueError("embedding batch generation returned the wrong number of sequences")
    return tuple(
        GenerationResult(
            response=tokenizer.decode(token_ids, skip_special_tokens=True),
            input_tokens=int(inputs_embeds.shape[1]),
            generated_tokens=len(token_ids),
            used_system_fallback=False,
        )
        for token_ids in output_ids
    )


def generate_embedding_response_record(
    *,
    model: Any,
    tokenizer: Any,
    schema_version: str,
    run_id: str,
    config_hash: str,
    sample_id: str,
    source: str,
    method: str,
    checkpoint: int,
    target_key: str,
    target_revision: str,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> ResponseRecord:
    """Generate a v1 response from a continuous embedding state.

    Reviewer-v2 has a deliberately narrower execution contract: embeddings
    must first be materialized to audited token IDs, then executed through
    :func:`generate_response_record`.  Rejecting v2 here prevents a caller
    from writing a valid-looking v2 response that never traversed projection.
    """
    if schema_version == "reviewer_eval.v2":
        raise ValueError(
            "continuous embedding generation does not support reviewer_eval.v2; "
            "materialize token IDs and use generate_response_record"
        )
    common = {
        "schema_version": schema_version,
        "run_id": run_id,
        "config_hash": config_hash,
        "sample_id": sample_id,
        "source": source,
        "method": method,
        "checkpoint": checkpoint,
        "target_key": target_key,
        "target_revision": target_revision,
        "prompt_hash": embedding_state_hash(inputs_embeds),
    }
    try:
        result = generate_from_embeddings(
            model,
            tokenizer,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        return ResponseRecord(
            **common,
            response=result.response,
            input_tokens=result.input_tokens,
            generated_tokens=result.generated_tokens,
            status=RecordStatus.complete,
            failure_kind=None,
            failure_reason=None,
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
        return ResponseRecord(
            **common,
            response="",
            input_tokens=0,
            generated_tokens=0,
            status=RecordStatus.failed,
            failure_kind=FailureKind.generation,
            failure_reason=f"target embedding generation error: {type(error).__name__}",
        )


def generate_one(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> GenerationResult:
    """Generate one greedy response through a tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        input_ids = _apply_chat_template(tokenizer, messages)
        used_system_fallback = False
    except (TypeError, ValueError) as error:
        if not _is_system_role_rejection(error):
            raise
        input_ids = _apply_chat_template(
            tokenizer,
            [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
        )
        used_system_fallback = True
    input_ids = move_to_model_input_device(input_ids, model)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    input_sequence = _first_sequence(input_ids)
    output_sequence = _first_sequence(output_ids)
    generated_ids = output_sequence[len(input_sequence) :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return GenerationResult(
        response=response,
        input_tokens=len(input_sequence),
        generated_tokens=len(generated_ids),
        used_system_fallback=used_system_fallback,
    )


def _generate_from_token_ids(
    model: Any, tokenizer: Any, *, input_ids: torch.Tensor, max_new_tokens: int
) -> GenerationResult:
    """Execute exact discrete IDs after the v2 record boundary has authorized them."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
        raise ValueError("v2 token inputs must have shape [1, tokens] and dtype long")
    input_ids = move_to_model_input_device(input_ids, model)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    input_sequence = _first_sequence(input_ids)
    output_sequence = _first_sequence(output_ids)
    generated_ids = output_sequence[len(input_sequence) :]
    return GenerationResult(
        response=tokenizer.decode(generated_ids, skip_special_tokens=True),
        input_tokens=len(input_sequence), generated_tokens=len(generated_ids),
        used_system_fallback=False,
    )


def generate_response_record(
    *,
    model: Any,
    tokenizer: Any,
    materialization: MaterializationRecord | V2MaterializationRecord,
    target_key: str,
    target_revision: str,
    max_new_tokens: int,
) -> ResponseRecord:
    """Generate one response only from an accepted materialization record.

    Failed materializations remain explicit failed response rows rather than
    being silently omitted, preserving matrix denominators for later analysis.
    """
    is_v2 = isinstance(materialization, V2MaterializationRecord)
    if is_v2:
        raise ValueError(
            "v2 token execution requires the local-assets stage; "
            "the public response API supports reviewer_eval.v1 only"
        )
    prompt_hash = hashlib.sha256(materialization.flat_prompt.encode("utf-8")).hexdigest()
    common = {
        "schema_version": materialization.schema_version,
        "run_id": materialization.run_id,
        "config_hash": materialization.config_hash,
        "sample_id": materialization.sample_id,
        "source": materialization.source,
        "method": materialization.method,
        "checkpoint": materialization.step if is_v2 else materialization.checkpoint,
        "target_key": target_key,
        "target_revision": target_revision,
        "prompt_hash": prompt_hash,
    }
    if materialization.status is not RecordStatus.complete:
        return ResponseRecord(
            **common,
            response="",
            input_tokens=0,
            generated_tokens=0,
            status=RecordStatus.failed,
            failure_kind=FailureKind.generation,
            failure_reason="materialization is not executable",
        )
    try:
        result = generate_one(
            model, tokenizer, system_prompt=materialization.system_prompt,
            user_prompt=materialization.user_prompt, max_new_tokens=max_new_tokens,
        )
        return ResponseRecord(
            **common,
            response=result.response,
            input_tokens=result.input_tokens,
            generated_tokens=result.generated_tokens,
            status=RecordStatus.complete,
            failure_kind=None,
            failure_reason=None,
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
        return ResponseRecord(
            **common,
            response="",
            input_tokens=0,
            generated_tokens=0,
            status=RecordStatus.failed,
            failure_kind=FailureKind.generation,
            failure_reason=f"target generation error: {type(error).__name__}",
        )


def _generate_v2_response_record(
    *,
    model: Any,
    tokenizer: Any,
    materialization: V2MaterializationRecord,
    target_key: str,
    target_revision: str,
    target_tokenizer_sha256: str,
    max_new_tokens: int,
) -> V2ResponseRecord:
    """Execute an already-authorized v2 materialization from a local runtime.

    This is intentionally private.  The pipeline's local-assets stage owns
    validation of the checkpoint ledger and target snapshot before reaching it.
    """
    if target_tokenizer_sha256 != materialization.surrogate_tokenizer_sha256:
        raise ValueError("v2 target tokenizer sha256 does not match materialization")
    prompt_hash = token_ids_sha256(materialization.complete_token_ids)
    common = {
        "schema_version": materialization.schema_version,
        "run_id": materialization.run_id,
        "config_hash": materialization.config_hash,
        "sample_id": materialization.sample_id,
        "source": materialization.source,
        "method": materialization.method,
        "checkpoint": materialization.step,
        "target_key": target_key,
        "target_revision": target_revision,
        "prompt_hash": prompt_hash,
        "branch": materialization.branch,
        "state_step": materialization.step,
        "transport": materialization.transport,
        "materialization_sha256": materialization.materialization_sha256,
        "target_tokenizer_sha256": target_tokenizer_sha256,
        "executed_token_ids_sha256": prompt_hash,
    }
    if materialization.status is not RecordStatus.complete:
        return V2ResponseRecord(
            **common, response="", input_tokens=0, generated_tokens=0,
            status=RecordStatus.failed, failure_kind=FailureKind.generation,
            failure_reason="materialization is not executable",
        )
    try:
        result = _generate_from_token_ids(
            model, tokenizer,
            input_ids=torch.tensor([materialization.complete_token_ids], dtype=torch.long),
            max_new_tokens=max_new_tokens,
        )
        return V2ResponseRecord(
            **common, response=result.response, input_tokens=result.input_tokens,
            generated_tokens=result.generated_tokens, status=RecordStatus.complete,
            failure_kind=None, failure_reason=None,
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
        return V2ResponseRecord(
            **common, response="", input_tokens=0, generated_tokens=0,
            status=RecordStatus.failed, failure_kind=FailureKind.generation,
            failure_reason=f"target generation error: {type(error).__name__}",
        )
