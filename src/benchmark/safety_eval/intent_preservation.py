"""Blinded human intent-preservation export and analysis primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .materialization_ablation import Branch, MaterializationPair, canonical_pair_key, index_pairs


class InitialLabel(StrEnum):
    preserved = "Preserved"
    not_preserved = "Not preserved"
    unsure = "Unsure"


class FinalLabel(StrEnum):
    preserved = "Preserved"
    not_preserved = "Not preserved"


class DriftReason(StrEnum):
    action_changed = "ACTION_CHANGED"
    target_changed = "TARGET_CHANGED"
    constraint_dropped = "CONSTRAINT_DROPPED"
    contradiction_added = "CONTRADICTION_ADDED"
    uninterpretable = "UNINTERPRETABLE"
    other = "OTHER"


@dataclass(frozen=True)
class AnnotationExport:
    blinded_csv: Path
    mapping_csv: Path
    manifest_json: Path
    annotation_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationMapping:
    annotation_id: str
    source: str
    sample_id: str
    branch: Branch
    optimization_checkpoint: int
    materialized_text_hash: str


@dataclass(frozen=True)
class RawLabel:
    annotation_id: str
    annotator_id: str
    initial_label: InitialLabel
    drift_reasons: tuple[DriftReason, ...]
    note: str


@dataclass(frozen=True)
class FinalIntentLabel:
    annotation_id: str
    final_label: FinalLabel
    drift_reasons: tuple[DriftReason, ...]
    adjudication_note: str


@dataclass(frozen=True)
class AnnotationAgreement:
    raw_agreement: float
    kappa: float | None
    denominator: int
    disagreements: int
    unsure_records: int
    adjudications_required: int


@dataclass(frozen=True)
class IPRRow:
    source: str
    branch: str
    preserved: int
    total: int
    rate: float


@dataclass(frozen=True)
class IntentAnalysis:
    agreement: AnnotationAgreement
    ipr: tuple[IPRRow, ...]
    drift_reason_counts: dict[str, int]
    judge_cross_tabs: dict[str, dict[str, dict[str, int]]]
    duplicate_materialized_prompts: int


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _annotation_id(pair: MaterializationPair, seed: int) -> str:
    key = "\x1f".join(map(str, canonical_pair_key(pair)))
    return "ann_" + hashlib.sha256(f"{seed}\x00{key}".encode("utf-8")).hexdigest()[:24]


def export_blinded_annotations(
    pairs: Iterable[MaterializationPair],
    *,
    seed: int,
    output_dir: Path,
) -> AnnotationExport:
    """Write deterministic blinded and private mapping artifacts."""
    complete = [pair for pair in index_pairs(pairs).values() if pair.status == "complete"]
    complete.sort(key=canonical_pair_key)
    if not complete:
        raise ValueError("annotation export requires at least one complete materialization pair")
    identifiers = {_annotation_id(pair, seed): pair for pair in complete}
    if len(identifiers) != len(complete):
        raise ValueError("annotation ID collision")
    ordered_ids = list(identifiers)
    random.Random(seed).shuffle(ordered_ids)

    blinded_path = output_dir / "intent_annotation_blinded.csv"
    mapping_path = output_dir / "intent_annotation_mapping.csv"
    manifest_path = output_dir / "intent_annotation_export.json"
    _write_csv(blinded_path, ("annotation_id", "reference_intent", "materialized_prompt"), (
        {
            "annotation_id": annotation_id,
            "reference_intent": identifiers[annotation_id].reference_intent,
            "materialized_prompt": identifiers[annotation_id].materialized_text,
        }
        for annotation_id in ordered_ids
    ))
    _write_csv(
        mapping_path,
        (
            "annotation_id", "source", "sample_id", "branch", "optimization_checkpoint",
            "materialized_text_hash",
        ),
        (
            {
                "annotation_id": annotation_id,
                "source": pair.source,
                "sample_id": pair.sample_id,
                "branch": pair.branch.value,
                "optimization_checkpoint": pair.optimization_checkpoint,
                "materialized_text_hash": hashlib.sha256(pair.materialized_text.encode("utf-8")).hexdigest(),
            }
            for annotation_id, pair in sorted(identifiers.items())
        ),
    )
    duplicate_count = len(complete) - len({pair.materialized_text for pair in complete})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "intent_annotation.v1",
                "seed": seed,
                "record_count": len(complete),
                "duplicate_materialized_prompts": duplicate_count,
                "blinded_sha256": hashlib.sha256(blinded_path.read_bytes()).hexdigest(),
                "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return AnnotationExport(blinded_path, mapping_path, manifest_path, tuple(ordered_ids))


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> Iterator[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"unexpected CSV columns in {path}")
        for line_number, row in enumerate(reader, start=2):
            if None in row.values():
                raise ValueError(f"malformed CSV row at line {line_number}")
            yield line_number, row


def read_mapping(path: Path) -> tuple[AnnotationMapping, ...]:
    expected = (
        "annotation_id", "source", "sample_id", "branch", "optimization_checkpoint",
        "materialized_text_hash",
    )
    rows: list[AnnotationMapping] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, Branch, int]] = set()
    for line_number, row in _read_csv(path, expected):
        try:
            mapping = AnnotationMapping(
                annotation_id=row["annotation_id"],
                source=row["source"],
                sample_id=row["sample_id"],
                branch=Branch(row["branch"]),
                optimization_checkpoint=int(row["optimization_checkpoint"]),
                materialized_text_hash=row["materialized_text_hash"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid annotation mapping at line {line_number}") from error
        if not mapping.annotation_id or len(mapping.materialized_text_hash) != 64:
            raise ValueError(f"invalid annotation mapping at line {line_number}")
        key = mapping.source, mapping.sample_id, mapping.branch, mapping.optimization_checkpoint
        if mapping.annotation_id in seen_ids or key in seen_keys:
            raise ValueError("duplicate annotation mapping")
        seen_ids.add(mapping.annotation_id)
        seen_keys.add(key)
        rows.append(mapping)
    if not rows:
        raise ValueError("annotation mapping is empty")
    return tuple(rows)


def _reasons(value: str, *, line_number: int) -> tuple[DriftReason, ...]:
    if not value.strip():
        return ()
    try:
        reasons = tuple(DriftReason(part.strip()) for part in value.split("|") if part.strip())
    except ValueError as error:
        raise ValueError(f"invalid drift reason at line {line_number}") from error
    if not reasons or len(set(reasons)) != len(reasons):
        raise ValueError(f"invalid drift reason at line {line_number}")
    return reasons


def _validate_reason_contract(
    label: InitialLabel | FinalLabel,
    reasons: tuple[DriftReason, ...],
    note: str,
    *,
    line_number: int,
) -> None:
    if label.value == FinalLabel.not_preserved.value:
        if not reasons:
            raise ValueError(f"Not preserved requires a drift reason at line {line_number}")
        if DriftReason.other in reasons and not note.strip():
            raise ValueError(f"OTHER requires a note at line {line_number}")
    elif reasons:
        raise ValueError(f"{label.value} cannot contain a drift reason at line {line_number}")


def read_raw_labels(
    path: Path,
    mapping_path: Path,
    *,
    expected_annotators: tuple[str, ...],
) -> tuple[RawLabel, ...]:
    mappings = read_mapping(mapping_path)
    annotation_ids = {row.annotation_id for row in mappings}
    if len(expected_annotators) < 2 or len(set(expected_annotators)) != len(expected_annotators):
        raise ValueError("expected annotators must contain at least two unique IDs")
    expected = ("annotation_id", "annotator_id", "initial_label", "drift_reason", "note")
    labels: list[RawLabel] = []
    seen: set[tuple[str, str]] = set()
    for line_number, row in _read_csv(path, expected):
        try:
            label = InitialLabel(row["initial_label"])
        except ValueError as error:
            raise ValueError(f"invalid initial label at line {line_number}") from error
        reasons = _reasons(row["drift_reason"], line_number=line_number)
        _validate_reason_contract(label, reasons, row["note"], line_number=line_number)
        key = row["annotation_id"], row["annotator_id"]
        if key[0] not in annotation_ids or key[1] not in expected_annotators or key in seen:
            raise ValueError(f"invalid or duplicate raw annotation at line {line_number}")
        seen.add(key)
        labels.append(RawLabel(key[0], key[1], label, reasons, row["note"]))
    required = {(annotation_id, annotator) for annotation_id in annotation_ids for annotator in expected_annotators}
    if seen != required:
        raise ValueError("raw annotation coverage is incomplete")
    return tuple(labels)


def read_final_labels(path: Path, mapping_path: Path) -> tuple[FinalIntentLabel, ...]:
    mappings = read_mapping(mapping_path)
    annotation_ids = {row.annotation_id for row in mappings}
    expected = ("annotation_id", "final_label", "drift_reason", "adjudication_note")
    labels: list[FinalIntentLabel] = []
    seen: set[str] = set()
    for line_number, row in _read_csv(path, expected):
        try:
            label = FinalLabel(row["final_label"])
        except ValueError as error:
            raise ValueError(f"invalid final label at line {line_number}") from error
        reasons = _reasons(row["drift_reason"], line_number=line_number)
        _validate_reason_contract(label, reasons, row["adjudication_note"], line_number=line_number)
        annotation_id = row["annotation_id"]
        if annotation_id not in annotation_ids or annotation_id in seen:
            raise ValueError(f"invalid or duplicate final annotation at line {line_number}")
        seen.add(annotation_id)
        labels.append(FinalIntentLabel(annotation_id, label, reasons, row["adjudication_note"]))
    if seen != annotation_ids:
        raise ValueError("final label coverage is incomplete")
    return tuple(labels)


def _agreement(labels: Iterable[RawLabel]) -> AnnotationAgreement:
    by_annotation: dict[str, list[RawLabel]] = defaultdict(list)
    annotators: set[str] = set()
    for label in labels:
        by_annotation[label.annotation_id].append(label)
        annotators.add(label.annotator_id)
    if len(annotators) != 2 or any(len(rows) != 2 for rows in by_annotation.values()):
        raise ValueError("Cohen's kappa requires exactly two complete annotators")
    ordered_annotators = tuple(sorted(annotators))
    pairs: list[tuple[InitialLabel, InitialLabel]] = []
    for rows in by_annotation.values():
        indexed = {row.annotator_id: row.initial_label for row in rows}
        pairs.append((indexed[ordered_annotators[0]], indexed[ordered_annotators[1]]))
    agreements = sum(left is right for left, right in pairs)
    unsure_records = sum(InitialLabel.unsure in pair for pair in pairs)
    adjudications_required = sum(left is not right or InitialLabel.unsure in (left, right) for left, right in pairs)
    denominator = len(pairs)
    observed = agreements / denominator
    expected = 0.0
    for label in InitialLabel:
        expected += (
            sum(left is label for left, _ in pairs) / denominator
        ) * (
            sum(right is label for _, right in pairs) / denominator
        )
    kappa = 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1.0 - expected)
    return AnnotationAgreement(
        observed,
        kappa,
        denominator,
        denominator - agreements,
        unsure_records,
        adjudications_required,
    )


def analyze_intent_labels(
    pairs: Iterable[MaterializationPair],
    mapping_path: Path,
    raw_labels: Iterable[RawLabel],
    final_labels: Iterable[FinalIntentLabel],
) -> IntentAnalysis:
    indexed_pairs = index_pairs(pairs)
    mappings = read_mapping(mapping_path)
    mapping_by_id = {row.annotation_id: row for row in mappings}
    final_by_id = {row.annotation_id: row for row in final_labels}
    if set(final_by_id) != set(mapping_by_id):
        raise ValueError("final labels do not match the annotation mapping")
    raw = tuple(raw_labels)
    raw_by_id: dict[str, list[RawLabel]] = defaultdict(list)
    for label in raw:
        raw_by_id[label.annotation_id].append(label)
    if set(raw_by_id) != set(mapping_by_id):
        raise ValueError("raw labels do not match the annotation mapping")
    agreement = _agreement(raw)
    for annotation_id, labels in raw_by_id.items():
        initial_labels = tuple(label.initial_label for label in labels)
        requires_adjudication = len(set(initial_labels)) != 1 or InitialLabel.unsure in initial_labels
        if requires_adjudication and not final_by_id[annotation_id].adjudication_note.strip():
            raise ValueError(f"adjudication note is required for annotation: {annotation_id}")
    joined: list[tuple[AnnotationMapping, MaterializationPair, FinalIntentLabel]] = []
    for annotation_id, mapping in mapping_by_id.items():
        key = mapping.source, mapping.sample_id, mapping.branch, mapping.optimization_checkpoint
        pair = indexed_pairs.get(key)
        if pair is None or pair.status != "complete":
            raise ValueError(f"annotation mapping references an unavailable pair: {key}")
        if hashlib.sha256(pair.materialized_text.encode("utf-8")).hexdigest() != mapping.materialized_text_hash:
            raise ValueError(f"materialized prompt hash mismatch: {annotation_id}")
        joined.append((mapping, pair, final_by_id[annotation_id]))

    cross_groups: dict[tuple[str, Branch], list[FinalIntentLabel]] = defaultdict(list)
    source_groups: dict[str, list[FinalIntentLabel]] = defaultdict(list)
    branch_groups: dict[Branch, list[FinalIntentLabel]] = defaultdict(list)
    for mapping, _, label in joined:
        cross_groups[(mapping.source, mapping.branch)].append(label)
        source_groups[mapping.source].append(label)
        branch_groups[mapping.branch].append(label)
    grouped: list[tuple[str, str, list[FinalIntentLabel]]] = []
    for (source, branch), labels in sorted(cross_groups.items(), key=lambda item: (item[0][0], item[0][1].value)):
        grouped.append((source, branch.value, labels))
    for source, labels in sorted(source_groups.items()):
        grouped.append((source, "All", labels))
    for branch in Branch:
        if branch in branch_groups:
            grouped.append(("All", branch.value, branch_groups[branch]))
    grouped.append(("Overall", "All", [label for _, _, label in joined]))
    ipr = tuple(
        IPRRow(
            source=source,
            branch=branch,
            preserved=sum(label.final_label is FinalLabel.preserved for label in labels),
            total=len(labels),
            rate=sum(label.final_label is FinalLabel.preserved for label in labels) / len(labels),
        )
        for source, branch, labels in grouped
    )
    reason_counts = Counter(
        reason.value
        for _, _, label in joined
        if label.final_label is FinalLabel.not_preserved
        for reason in label.drift_reasons
    )
    judge_keys = sorted({key for _, pair, _ in joined for key in pair.judgments})
    cross_tabs: dict[str, dict[str, dict[str, int]]] = {}
    for judge_key in judge_keys:
        table = {
            FinalLabel.preserved.value: {"safe": 0, "unsafe": 0},
            FinalLabel.not_preserved.value: {"safe": 0, "unsafe": 0},
        }
        for _, pair, label in joined:
            judgment = pair.judgments.get(judge_key)
            if judgment is None:
                raise ValueError(f"missing materialized judge label for {judge_key}")
            outcome = "unsafe" if judgment.materialized else "safe"
            table[label.final_label.value][outcome] += 1
        cross_tabs[judge_key] = table
    duplicate_count = len(joined) - len({pair.materialized_text for _, pair, _ in joined})
    return IntentAnalysis(
        agreement=agreement,
        ipr=ipr,
        drift_reason_counts=dict(sorted(reason_counts.items())),
        judge_cross_tabs=cross_tabs,
        duplicate_materialized_prompts=duplicate_count,
    )
