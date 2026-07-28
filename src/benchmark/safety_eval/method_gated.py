"""Fail-closed method-level gates over immutable safety-evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_jsonl
from .pipeline import judge_response_records
from .schema import JudgmentRecord, MaterializationRecord, RecordStatus, ResponseRecord


class MethodGateError(RuntimeError):
    """Raised when a source-method cell is not valid for selected-matrix use."""


@dataclass(frozen=True)
class SourceMethodEvidence:
    """Validated parent records for one selected source-method cell."""

    source: str
    method: str
    responses: tuple[ResponseRecord, ...]
    response_count: int
    primary_judgment_count: int


@dataclass(frozen=True)
class MethodGateEvidence:
    """Validated parent and selected-ledger counts for one complete method."""

    method: str
    source_count: int
    response_count: int
    primary_judgment_count: int
    secondary_judgment_count: int


def _manifest_ids(parent_root: Path, source: str, samples_per_source: int) -> set[str]:
    rows = read_jsonl(parent_root / "manifests" / f"controlled_{source}.jsonl")
    sample_ids = [row.get("example_id") for row in rows]
    if len(sample_ids) != samples_per_source or any(not isinstance(value, str) or not value for value in sample_ids):
        raise MethodGateError("controlled manifest is incomplete")
    if len(set(sample_ids)) != samples_per_source:
        raise MethodGateError("controlled manifest contains duplicate sample IDs")
    return set(sample_ids)


def _final_materializations(parent_root: Path, *, source: str, method: str, sample_ids: set[str]) -> None:
    checkpoint = 0 if method == "init" else 100
    rows = [
        MaterializationRecord.model_validate(row)
        for row in read_jsonl(parent_root / "optimization" / source / method / "materialization.jsonl")
    ]
    selected = [row for row in rows if row.checkpoint == checkpoint]
    by_id = {row.sample_id: row for row in selected}
    if len(by_id) != len(selected) or set(by_id) != sample_ids:
        raise MethodGateError("final materializations are incomplete")
    if any(row.status is not RecordStatus.complete or not row.intent_preserved for row in selected):
        raise MethodGateError("final materialization is not executable")


def _responses(
    parent_root: Path,
    *,
    source: str,
    method: str,
    target_key: str,
    sample_ids: set[str],
) -> tuple[ResponseRecord, ...]:
    checkpoint = 0 if method == "init" else 100
    rows = [
        ResponseRecord.model_validate(row)
        for row in read_jsonl(parent_root / "responses" / target_key / source / method / "records.jsonl")
    ]
    selected = [row for row in rows if row.checkpoint == checkpoint]
    by_id = {row.sample_id: row for row in selected}
    if len(by_id) != len(selected) or set(by_id) != sample_ids:
        raise MethodGateError("target responses are incomplete")
    if any(row.status is not RecordStatus.complete for row in selected):
        raise MethodGateError("target response is not executable")
    return tuple(by_id[sample_id] for sample_id in sorted(sample_ids))


def _primary_judgments(
    parent_root: Path,
    *,
    source: str,
    method: str,
    target_key: str,
    primary_key: str,
    sample_ids: set[str],
    thresholds: tuple[float, ...],
) -> int:
    rows = [
        JudgmentRecord.model_validate(row)
        for row in read_jsonl(parent_root / "judgments" / primary_key / target_key / source / method / "records.jsonl")
    ]
    wanted = {(sample_id, threshold) for sample_id in sample_ids for threshold in thresholds}
    selected = [row for row in rows if (row.sample_id, row.threshold) in wanted]
    by_key = {(row.sample_id, row.threshold): row for row in selected}
    if len(by_key) != len(selected) or set(by_key) != wanted:
        raise MethodGateError("primary judgments are incomplete")
    if any(row.status is RecordStatus.failed for row in selected):
        raise MethodGateError("failed primary judgment")
    if any(row.status is not RecordStatus.complete for row in selected):
        raise MethodGateError("primary judgment is not complete")
    return len(selected)


def validate_source_method(
    *,
    parent_root: Path,
    source: str,
    method: str,
    target_key: str,
    samples_per_source: int,
    primary_key: str,
    thresholds: tuple[float, ...],
) -> SourceMethodEvidence:
    """Return parent responses only after all selected upstream stages validate."""
    if not source or not method or not target_key:
        raise MethodGateError("source, method, and target key must be non-empty")
    if samples_per_source < 1 or not thresholds or len(set(thresholds)) != len(thresholds):
        raise MethodGateError("selected matrix dimensions are invalid")
    sample_ids = _manifest_ids(parent_root, source, samples_per_source)
    _final_materializations(parent_root, source=source, method=method, sample_ids=sample_ids)
    responses = _responses(
        parent_root,
        source=source,
        method=method,
        target_key=target_key,
        sample_ids=sample_ids,
    )
    primary_count = _primary_judgments(
        parent_root,
        source=source,
        method=method,
        target_key=target_key,
        primary_key=primary_key,
        sample_ids=sample_ids,
        thresholds=thresholds,
    )
    return SourceMethodEvidence(
        source=source,
        method=method,
        responses=responses,
        response_count=len(responses),
        primary_judgment_count=primary_count,
    )


def probe_compatibility_judge(judge: Any) -> None:
    """Require the compatibility judge to complete one content-free contract probe."""
    record = JudgmentRecord.model_validate(
        judge.judge_response(
            run_id="method-gate:probe",
            config_hash="0" * 64,
            sample_id="synthetic:neutral",
            source="synthetic",
            method="init",
            checkpoint=0,
            target_key="synthetic",
            target_revision="synthetic",
            response_text="This is a neutral, harmless assistant response.",
            threshold=0.5,
        )
    )
    if record.status is not RecordStatus.complete or type(record.unsafe_label) is not bool:
        raise MethodGateError("compatibility probe did not satisfy the JSON contract")


def _secondary_judgment_count(
    selection_root: Path,
    *,
    judge_key: str,
    target_key: str,
    source: str,
    method: str,
    sample_ids: set[str],
    thresholds: tuple[float, ...],
) -> int:
    rows = [
        JudgmentRecord.model_validate(row)
        for row in read_jsonl(selection_root / "judgments" / judge_key / target_key / source / method / "records.jsonl")
    ]
    wanted = {(sample_id, threshold) for sample_id in sample_ids for threshold in thresholds}
    selected = [row for row in rows if (row.sample_id, row.threshold) in wanted]
    by_key = {(row.sample_id, row.threshold): row for row in selected}
    if len(by_key) != len(selected) or set(by_key) != wanted:
        raise MethodGateError("secondary judgments are incomplete")
    if any(row.status is RecordStatus.failed for row in selected):
        raise MethodGateError("failed secondary judgment")
    if any(row.status is not RecordStatus.complete for row in selected):
        raise MethodGateError("secondary judgment is not complete")
    return len(selected)


def _run_method_gate(
    *,
    parent_root: Path,
    selection_root: Path,
    sources: tuple[str, ...],
    method: str,
    target_key: str,
    samples_per_source: int,
    primary_key: str,
    thresholds: tuple[float, ...],
    judge: Any,
    selection_hash: str,
) -> MethodGateEvidence:
    """Validate one method and write only its secondary judgments to the selection root."""
    if len(sources) != len(set(sources)) or not sources:
        raise MethodGateError("selected sources must be non-empty and unique")
    if not isinstance(selection_hash, str) or len(selection_hash) != 64:
        raise MethodGateError("selection hash is invalid")
    source_evidence = tuple(
        validate_source_method(
            parent_root=parent_root,
            source=source,
            method=method,
            target_key=target_key,
            samples_per_source=samples_per_source,
            primary_key=primary_key,
            thresholds=thresholds,
        )
        for source in sources
    )
    probe_compatibility_judge(judge)
    for evidence in source_evidence:
        for threshold in thresholds:
            judge_response_records(selection_root, (response.model_dump(mode="json") for response in evidence.responses), judge=judge, threshold=threshold)
    secondary_count = sum(
        _secondary_judgment_count(
            selection_root,
            judge_key=str(judge.key),
            target_key=target_key,
            source=evidence.source,
            method=method,
            sample_ids={response.sample_id for response in evidence.responses},
            thresholds=thresholds,
        )
        for evidence in source_evidence
    )
    return MethodGateEvidence(
        method=method,
        source_count=len(source_evidence),
        response_count=sum(item.response_count for item in source_evidence),
        primary_judgment_count=sum(item.primary_judgment_count for item in source_evidence),
        secondary_judgment_count=secondary_count,
    )


def run_method_gate(
    *,
    parent_root: Path,
    selection_root: Path,
    sources: tuple[str, ...],
    method: str,
    target_key: str,
    samples_per_source: int,
    primary_key: str,
    thresholds: tuple[float, ...],
    judge: Any,
    selection_hash: str,
) -> MethodGateEvidence:
    """Execute one fail-closed method gate and persist its terminal status."""
    try:
        evidence = _run_method_gate(
            parent_root=parent_root,
            selection_root=selection_root,
            sources=sources,
            method=method,
            target_key=target_key,
            samples_per_source=samples_per_source,
            primary_key=primary_key,
            thresholds=thresholds,
            judge=judge,
            selection_hash=selection_hash,
        )
    except MethodGateError as error:
        atomic_write_json(
            selection_root / "failures" / f"{method}.failed.json",
            {"failure": str(error), "method": method},
        )
        raise
    atomic_write_json(
        selection_root / "methods" / f"{method}.complete.json",
        {
            "method": evidence.method,
            "primary_judgment_count": evidence.primary_judgment_count,
            "response_count": evidence.response_count,
            "secondary_judgment_count": evidence.secondary_judgment_count,
            "selection_hash": selection_hash,
            "source_count": evidence.source_count,
        },
    )
    return evidence
