"""Strict, serializable records for safety-evaluation artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .io import canonical_hash


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RecordStatus(StrEnum):
    complete = "complete"
    failed = "failed"
    frozen_pdf = "frozen_pdf"
    skipped_exact_frozen = "skipped_exact_frozen"


class FailureKind(StrEnum):
    oom = "oom"
    tokenizer = "tokenizer"
    optimization = "optimization"
    materialization = "materialization"
    semantic_filter = "semantic_filter"
    generation = "generation"
    judge = "judge"
    source_data = "source_data"
    compatibility = "compatibility"


def stable_id(prefix: str, payload: object) -> str:
    """Return a compact, reproducible identifier for a canonical payload."""
    return f"{prefix}:{canonical_hash(payload)[:20]}"


class StrictRecord(BaseModel):
    """Base contract for persisted records.

    ``model_copy(update=...)`` is supported for identity and fixture
    construction, but updates must satisfy the same schema validation as data
    loaded from a persisted record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)

        payload = self.model_dump(mode="json", round_trip=True)
        payload.update(update)
        return type(self).model_validate(payload)

    @model_validator(mode="after")
    def validate_failure_fields(self) -> "StrictRecord":
        status = getattr(self, "status", None)
        failure_kind = getattr(self, "failure_kind", None)
        failure_reason = getattr(self, "failure_reason", None)
        has_failure_kind = failure_kind is not None
        has_failure_reason = failure_reason is not None and bool(failure_reason.strip())

        if status is RecordStatus.failed:
            if not has_failure_kind or not has_failure_reason:
                raise ValueError("failed records require failure_kind and failure_reason")
        elif status is RecordStatus.complete:
            if has_failure_kind or failure_reason is not None:
                raise ValueError("complete records cannot include failure fields")
        return self


class BenchmarkExample(StrictRecord):
    example_id: str
    source: str
    source_file: str
    source_row: int
    source_sha256: Sha256
    intent: str
    attack_text: str
    target_text: str | None
    source_risk_label: str | None
    source_attack_label: str
    risk_category: str
    threat_domain: str
    attack_type: str
    language: str
    selection_stratum: str
    selection_seed: int
    prompt_sha256: Sha256
    preprocessing: tuple[str, ...]


class ManifestHeader(StrictRecord):
    schema_version: str
    manifest_hash: Sha256
    source: str
    source_file_sha256: Sha256
    config_hash: Sha256
    record_count: int
    ordered_example_ids: tuple[str, ...]


class CellKey(StrictRecord):
    dataset_source: str
    sample_manifest_hash: Sha256
    optimization_method: str
    optimization_budget: str
    surrogate_model_revision: str
    target_model_revision: str
    decoding_config_hash: Sha256
    judge_revision: str
    judge_threshold: float

    @property
    def cell_id(self) -> str:
        return stable_id("cell", self.model_dump(mode="json"))


class ComputeCounters(StrictRecord):
    updates: int = 0
    forward_passes: int = 0
    backward_passes: int = 0
    hvp_calls: int = 0
    candidates_attempted: int = 0
    candidates_accepted: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    judged_tokens: int = 0
    wall_seconds: float = 0.0
    peak_gpu_bytes: int = 0


class OptimizationRecord(StrictRecord):
    schema_version: str
    run_id: str
    config_hash: Sha256
    git_revision: str
    cell_id: str
    sample_id: str
    source: str
    method: str
    checkpoint: int
    random_seed: int
    status: RecordStatus
    failure_kind: FailureKind | None
    failure_reason: str | None
    state_path: str | None
    representation: str
    attack_loss: float | None
    fol: float | None
    internal_margin: float | None
    materialized_prompt: str | None
    counters: ComputeCounters


class MaterializationRecord(StrictRecord):
    schema_version: str
    run_id: str
    config_hash: Sha256
    sample_id: str
    source: str
    method: str
    checkpoint: int
    system_prompt: str
    user_prompt: str
    flat_prompt: str
    prefix_token_ids: tuple[int, ...]
    seed_token_ids: tuple[int, ...]
    prefix_projection_cosine: float | None
    seed_projection_cosine: float | None
    semantic_similarity_before: float
    semantic_similarity_after: float
    category_before: str
    category_after: str
    intent_preserved: bool
    projection_attack_score_before: float | None
    projection_attack_score_after: float | None
    status: RecordStatus
    failure_kind: FailureKind | None
    failure_reason: str | None


class ResponseRecord(StrictRecord):
    schema_version: str
    run_id: str
    config_hash: Sha256
    sample_id: str
    source: str
    method: str
    checkpoint: int
    target_key: str
    target_revision: str
    prompt_hash: Sha256
    response: str
    input_tokens: int
    generated_tokens: int
    status: RecordStatus
    failure_kind: FailureKind | None
    failure_reason: str | None


class JudgmentRecord(StrictRecord):
    schema_version: str
    run_id: str
    config_hash: Sha256
    sample_id: str
    source: str
    method: str
    checkpoint: int
    target_key: str
    judge_key: str
    judge_revision: str
    unsafe_score: float
    unsafe_label: bool
    refusal_label: bool | None
    threshold: float
    raw_output: str
    status: RecordStatus
    failure_kind: FailureKind | None
    failure_reason: str | None
