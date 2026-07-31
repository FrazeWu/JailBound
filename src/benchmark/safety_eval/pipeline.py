"""Content-free, resumable persistence for post-optimization stages."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch

from .io import JsonlLedger, read_jsonl
from .materialization import materialize_checkpoint
from .generation import generate_response_record
from .schema import BenchmarkExample, FailureKind, JudgmentRecord, MaterializationRecord, OptimizationRecord, RecordStatus, ResponseRecord


@dataclass(frozen=True)
class StageSummary:
    """Counts emitted by one resumable stage write."""

    selected_records: int
    written_records: int
    failed_records: int


def failed_optimization_materialization(
    optimization: OptimizationRecord,
    *,
    category: str,
) -> MaterializationRecord:
    """Carry a terminal optimization failure into the materialization ledger."""

    if optimization.status is not RecordStatus.failed:
        raise ValueError("only failed optimization records can be carried forward")
    if not isinstance(category, str) or not category:
        raise ValueError("materialization category must be non-empty")
    return MaterializationRecord(
        schema_version=optimization.schema_version,
        run_id=optimization.run_id,
        config_hash=optimization.config_hash,
        sample_id=optimization.sample_id,
        source=optimization.source,
        method=optimization.method,
        checkpoint=optimization.checkpoint,
        system_prompt="",
        user_prompt="",
        flat_prompt="",
        prefix_token_ids=(),
        seed_token_ids=(),
        prefix_projection_cosine=None,
        seed_projection_cosine=None,
        semantic_similarity_before=0.0,
        semantic_similarity_after=0.0,
        category_before=category,
        category_after=category,
        intent_preserved=False,
        projection_attack_score_before=None,
        projection_attack_score_after=None,
        status=RecordStatus.failed,
        failure_kind=FailureKind.optimization,
        failure_reason="upstream optimization record was not executable",
    )


def select_final_optimization_records(
    records: Iterable[OptimizationRecord],
) -> tuple[OptimizationRecord, ...]:
    """Select the only checkpoints eligible for target generation.

    Init is an unoptimized reference at checkpoint zero.  Every other method
    is represented by its terminal checkpoint, including typed terminal
    failures, so target-stage denominators remain fixed.
    """

    selected: dict[tuple[str, str, str], OptimizationRecord] = {}
    for record in records:
        expected_checkpoint = 0 if record.method == "init" else 100
        if record.checkpoint != expected_checkpoint:
            continue
        key = (record.source, record.method, record.sample_id)
        if key in selected:
            raise ValueError("multiple final optimization records for one source, method, and sample")
        selected[key] = record
    return tuple(selected[key] for key in sorted(selected))


def write_materialization_records(
    output_root: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> StageSummary:
    """Persist materialization outcomes beside their immutable optimizer states."""

    selected = written = failed = 0
    ledgers: dict[tuple[str, str], JsonlLedger] = {}
    root = Path(output_root) / "optimization"
    for row in rows:
        selected += 1
        sample_id, source, method, checkpoint, status = _required_row_fields(row)
        if status == "failed":
            failed += 1
        key = (source, method)
        ledger = ledgers.setdefault(
            key,
            JsonlLedger(root / source / method / "materialization.jsonl", key_fields=("sample_id", "checkpoint")),
        )
        normalized = dict(row)
        normalized.update(
            sample_id=sample_id,
            source=source,
            method=method,
            checkpoint=checkpoint,
            status=status,
        )
        if ledger.append_once(normalized):
            written += 1
    return StageSummary(selected, written, failed)


def materialize_optimization_record(
    optimization: OptimizationRecord,
    *,
    example: BenchmarkExample,
    vocabulary_embeddings: torch.Tensor,
    tokenizer: Any,
    semantic_similarity: Callable[[str, str], float],
    semantic_threshold: float,
) -> MaterializationRecord:
    """Recover one optimizer state and apply one frozen semantic decision."""

    if (optimization.sample_id, optimization.source) != (example.example_id, example.source):
        raise ValueError("optimization record does not match its immutable manifest example")
    if optimization.status is RecordStatus.failed:
        return failed_optimization_materialization(optimization, category=example.risk_category)
    if optimization.status is not RecordStatus.complete or not optimization.state_path:
        return _materialization_failure(optimization, category=example.risk_category, reason="missing optimizer state")
    try:
        payload = torch.load(Path(optimization.state_path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("state payload must be a mapping")
        provisional = materialize_checkpoint(
            state_payload=payload,
            vocabulary_embeddings=vocabulary_embeddings,
            tokenizer=tokenizer,
            schema_version=optimization.schema_version,
            run_id=optimization.run_id,
            config_hash=optimization.config_hash,
            sample_id=optimization.sample_id,
            source=optimization.source,
            method=optimization.method,
            checkpoint=optimization.checkpoint,
            original_prompt=example.attack_text,
            category=example.risk_category,
            semantic_similarity=1.0,
            semantic_threshold=0.0,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return _materialization_failure(
            optimization,
            category=example.risk_category,
            reason=f"state materialization error: {type(error).__name__}",
        )
    if provisional.status is RecordStatus.failed:
        return provisional
    score = float(semantic_similarity(example.attack_text, provisional.flat_prompt))
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        return _materialization_failure(optimization, category=example.risk_category, reason="invalid semantic similarity")
    score = max(0.0, score)
    if score < semantic_threshold:
        return provisional.model_copy(
            update={
                "semantic_similarity_after": score,
                "intent_preserved": False,
                "status": RecordStatus.failed,
                "failure_kind": FailureKind.semantic_filter,
                "failure_reason": "below semantic threshold",
            }
        )
    return provisional.model_copy(update={"semantic_similarity_after": score})


def materialize_records_from_disk(
    output_root: str | Path,
    *,
    vocabulary_embeddings: torch.Tensor,
    tokenizer: Any,
    semantic_similarity: Callable[[str, str], float],
    semantic_threshold: float,
    final_only: bool,
) -> StageSummary:
    """Join immutable manifests to optimizer ledgers and persist materializations."""

    root = Path(output_root)
    examples: dict[str, BenchmarkExample] = {}
    for path in sorted((root / "manifests").glob("controlled_*.jsonl")):
        for payload in read_jsonl(path):
            example = BenchmarkExample.model_validate(payload)
            if example.example_id in examples:
                raise ValueError("duplicate immutable manifest example id")
            examples[example.example_id] = example
    if not examples:
        raise ValueError("no immutable controlled manifests found")

    optimizations: list[OptimizationRecord] = []
    for path in sorted((root / "optimization").glob("*/*/records.jsonl")):
        optimizations.extend(OptimizationRecord.model_validate(payload) for payload in read_jsonl(path))
    selected = select_final_optimization_records(optimizations) if final_only else tuple(
        sorted(optimizations, key=lambda row: (row.source, row.method, row.sample_id, row.checkpoint))
    )
    records: list[Mapping[str, Any]] = []
    for optimization in selected:
        example = examples.get(optimization.sample_id)
        if example is None:
            raise ValueError("optimization record references an unknown manifest example")
        record = materialize_optimization_record(
            optimization,
            example=example,
            vocabulary_embeddings=vocabulary_embeddings,
            tokenizer=tokenizer,
            semantic_similarity=semantic_similarity,
            semantic_threshold=semantic_threshold,
        )
        records.append(record.model_dump(mode="json"))
    return write_materialization_records(root, records)


def generate_materialized_records(
    output_root: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    target_key: str,
    target_revision: str,
    max_new_tokens: int,
) -> StageSummary:
    """Generate one greedy response per final materialization, resumably."""

    selected = written = failed = 0
    ledgers: dict[tuple[str, str], JsonlLedger] = {}
    for row in rows:
        selected += 1
        materialization = MaterializationRecord.model_validate(row)
        key = (materialization.source, materialization.method)
        ledger = ledgers.setdefault(
            key,
            JsonlLedger(
                Path(output_root)
                / "responses"
                / _path_component(target_key, field="target_key")
                / materialization.source
                / materialization.method
                / "records.jsonl",
                key_fields=("sample_id", "checkpoint"),
            ),
        )
        if ledger.contains_key(
            {
                "sample_id": materialization.sample_id,
                "checkpoint": materialization.checkpoint,
            }
        ):
            continue
        response = generate_response_record(
            model=model, tokenizer=tokenizer, materialization=materialization,
            target_key=target_key, target_revision=target_revision, max_new_tokens=max_new_tokens,
        )
        if response.status is RecordStatus.failed:
            failed += 1
        if ledger.append_once(response.model_dump(mode="json")):
            written += 1
    return StageSummary(selected, written, failed)


def judge_response_records(
    output_root: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    judge: Any,
    threshold: float,
) -> StageSummary:
    """Persist one terminal judgment per response and threshold."""

    selected = written = failed = 0
    judge_key = _path_component(str(getattr(judge, "key")), field="judge_key")
    ledgers: dict[tuple[str, str, str], JsonlLedger] = {}
    for row in rows:
        selected += 1
        response = ResponseRecord.model_validate(row)
        key = (response.target_key, response.source, response.method)
        ledger = ledgers.setdefault(
            key,
            JsonlLedger(
                Path(output_root)
                / "judgments"
                / judge_key
                / _path_component(response.target_key, field="target_key")
                / response.source
                / response.method
                / "records.jsonl",
                key_fields=("sample_id", "checkpoint", "threshold"),
            ),
        )
        if ledger.contains_key(
            {
                "sample_id": response.sample_id,
                "checkpoint": response.checkpoint,
                "threshold": threshold,
            }
        ):
            continue
        if response.status is RecordStatus.failed:
            record = JudgmentRecord(
                schema_version=response.schema_version, run_id=response.run_id, config_hash=response.config_hash,
                sample_id=response.sample_id, source=response.source, method=response.method, checkpoint=response.checkpoint,
                target_key=response.target_key, judge_key=str(getattr(judge, "key")),
                judge_revision=str(getattr(judge, "revision")), unsafe_score=0.0, unsafe_label=False,
                refusal_label=None, threshold=threshold, raw_output="", status=RecordStatus.failed,
                failure_kind=FailureKind.judge, failure_reason="response generation was not executable",
            )
        else:
            result = judge.judge_response(
                run_id=response.run_id, config_hash=response.config_hash, sample_id=response.sample_id,
                source=response.source, method=response.method, checkpoint=response.checkpoint,
                target_key=response.target_key, target_revision=response.target_revision,
                response_text=response.response, threshold=threshold,
            )
            record = JudgmentRecord.model_validate(result)
        if record.status is RecordStatus.failed:
            failed += 1
        if ledger.append_once(record.model_dump(mode="json")):
            written += 1
    return StageSummary(selected, written, failed)


def _materialization_failure(
    optimization: OptimizationRecord,
    *,
    category: str,
    reason: str,
) -> MaterializationRecord:
    return MaterializationRecord(
        schema_version=optimization.schema_version,
        run_id=optimization.run_id,
        config_hash=optimization.config_hash,
        sample_id=optimization.sample_id,
        source=optimization.source,
        method=optimization.method,
        checkpoint=optimization.checkpoint,
        system_prompt="",
        user_prompt="",
        flat_prompt="",
        prefix_token_ids=(),
        seed_token_ids=(),
        prefix_projection_cosine=None,
        seed_projection_cosine=None,
        semantic_similarity_before=0.0,
        semantic_similarity_after=0.0,
        category_before=category,
        category_after=category,
        intent_preserved=False,
        projection_attack_score_before=None,
        projection_attack_score_after=None,
        status=RecordStatus.failed,
        failure_kind=FailureKind.materialization,
        failure_reason=reason,
    )


def _path_component(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{field} must be a single safe path component")
    return value


def _required_row_fields(row: Mapping[str, object]) -> tuple[str, str, str, int, str]:
    sample_id = row.get("sample_id")
    source = _path_component(row.get("source"), field="source")
    method = _path_component(row.get("method"), field="method")
    checkpoint = row.get("checkpoint")
    status = row.get("status")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("stage rows require a non-empty sample_id")
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint < 0:
        raise ValueError("stage rows require a non-negative integer checkpoint")
    if status not in {"complete", "failed"}:
        raise ValueError("stage rows require a terminal status")
    return sample_id, source, method, checkpoint, status


def write_stage_records(
    output_root: str | Path,
    *,
    stage: str,
    rows: Iterable[Mapping[str, Any]],
    partition_fields: tuple[str, ...] = (),
    key_fields: tuple[str, ...] = ("sample_id", "checkpoint"),
) -> StageSummary:
    """Append stage rows once per source, method, sample, and checkpoint.

    No model text is interpreted here.  The helper exists so materialization,
    response, and judgment stages share one strict, crash-safe resume rule.
    """

    stage_name = _path_component(stage, field="stage")
    if not key_fields or len(set(key_fields)) != len(key_fields):
        raise ValueError("stage key_fields must be unique and non-empty")
    if len(set(partition_fields)) != len(partition_fields):
        raise ValueError("stage partition_fields must be unique")
    for field in (*partition_fields, *key_fields):
        _path_component(field, field="field name")
    selected = written = failed = 0
    ledgers: dict[tuple[str, ...], JsonlLedger] = {}
    root = Path(output_root) / stage_name
    for row in rows:
        selected += 1
        sample_id, source, method, checkpoint, status = _required_row_fields(row)
        if status == "failed":
            failed += 1
        partition = tuple(_path_component(row.get(field), field=field) for field in partition_fields)
        key = partition + (source, method)
        ledger = ledgers.setdefault(
            key,
            JsonlLedger(root.joinpath(*partition, source, method, "records.jsonl"), key_fields=key_fields),
        )
        normalized = dict(row)
        normalized.update(
            sample_id=sample_id,
            source=source,
            method=method,
            checkpoint=checkpoint,
            status=status,
        )
        if ledger.append_once(normalized):
            written += 1
    return StageSummary(selected, written, failed)
