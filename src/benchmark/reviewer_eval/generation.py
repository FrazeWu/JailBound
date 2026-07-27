"""Local target-generation helpers with no runner dependencies."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .schema import FailureKind, MaterializationRecord, RecordStatus, ResponseRecord


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


def generate_response_record(
    *,
    model: Any,
    tokenizer: Any,
    materialization: MaterializationRecord,
    target_key: str,
    target_revision: str,
    max_new_tokens: int,
) -> ResponseRecord:
    """Generate one response only from an accepted materialization record.

    Failed materializations remain explicit failed response rows rather than
    being silently omitted, preserving matrix denominators for later analysis.
    """
    prompt_hash = hashlib.sha256(materialization.flat_prompt.encode("utf-8")).hexdigest()
    common = {
        "schema_version": materialization.schema_version,
        "run_id": materialization.run_id,
        "config_hash": materialization.config_hash,
        "sample_id": materialization.sample_id,
        "source": materialization.source,
        "method": materialization.method,
        "checkpoint": materialization.checkpoint,
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
            model,
            tokenizer,
            system_prompt=materialization.system_prompt,
            user_prompt=materialization.user_prompt,
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
            failure_reason=f"target generation error: {type(error).__name__}",
        )
