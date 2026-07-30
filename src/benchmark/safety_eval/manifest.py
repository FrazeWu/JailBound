"""Deterministic sampling contracts for controlled safety experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Protocol, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io import atomic_write_json, canonical_hash, canonical_json
from .schema import BenchmarkExample, FailureKind, ManifestHeader, V2BenchmarkExample
from .datasets import RawExample
from .semantic import Encoder, MappingDecision, map_raw_example
from .span_annotation import FrozenSpanAnnotation, SpanAnnotationError


@dataclass(frozen=True)
class SelectionReport:
    duplicate_count: int
    eligible_count: int
    selected_count: int


@dataclass(frozen=True)
class AnnotationFailure:
    """Content-free audit record for one rejected v2 candidate."""

    schema_version: str
    example_id: str
    source: str
    source_file: str
    source_row: int
    source_sha256: str
    prompt_sha256: str
    intent_sha256: str
    failure_kind: FailureKind
    failure_reason: str

    def __post_init__(self) -> None:
        expected_codes = {
            FailureKind.annotation: "annotation_error",
            FailureKind.transport: "transport_error",
        }
        if expected_codes.get(self.failure_kind) != self.failure_reason:
            raise ValueError(
                "annotation failures require the stable failure code for their kind"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "source": self.source,
            "source_file": self.source_file,
            "source_row": self.source_row,
            "source_sha256": self.source_sha256,
            "prompt_sha256": self.prompt_sha256,
            "intent_sha256": self.intent_sha256,
            "failure_kind": self.failure_kind.value,
            "failure_reason": self.failure_reason,
        }


class CandidateAnnotator(Protocol):
    def annotate(
        self,
        prompt: str,
        *,
        seed_intent: str,
        source_hints: Mapping[str, object],
    ) -> FrozenSpanAnnotation: ...


@dataclass(frozen=True)
class FolCandidate:
    sample_id: str
    source: str
    fol: float
    risk_category: str
    initial_label: bool
    attack_loss: float
    token_length: int
    perplexity: float


@dataclass(frozen=True)
class FolSplit:
    low: tuple[FolCandidate, ...]
    middle: tuple[FolCandidate, ...]
    high: tuple[FolCandidate, ...]
    status: str
    unmatched: tuple[str, ...] = ()
    matching_caliper: float | None = None
    matching_distances: tuple[float, ...] = ()


def _atomic_write_jsonl(path: Path, rows: Sequence[BenchmarkExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in rows).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def write_controlled_manifest(
    output_root: str | Path, source: str, records: Sequence[BenchmarkExample], *,
    source_file_sha256: str, config_hash: str,
) -> ManifestHeader:
    """Write once, then require byte-for-byte logical identity on reruns."""
    manifests = Path(output_root) / "manifests"
    path = manifests / f"controlled_{source}.jsonl"
    header_path = manifests / f"controlled_{source}.header.json"
    ordered = tuple(sorted(records, key=lambda row: row.example_id))
    payloads = [row.model_dump(mode="json") for row in ordered]
    header = ManifestHeader(
        schema_version="reviewer_eval.v1", manifest_hash=canonical_hash(payloads), source=source,
        source_file_sha256=source_file_sha256, config_hash=config_hash,
        record_count=len(ordered), ordered_example_ids=tuple(row.example_id for row in ordered),
    )
    if path.exists() or header_path.exists():
        if not path.exists() or not header_path.exists():
            raise ValueError("incomplete immutable manifest")
        existing_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        existing_header = ManifestHeader.model_validate(json.loads(header_path.read_text(encoding="utf-8")))
        if existing_rows != payloads or existing_header != header:
            raise ValueError("immutable manifest differs from existing output")
        return header
    _atomic_write_jsonl(path, ordered)
    atomic_write_json(header_path, header.model_dump(mode="json"))
    return header


def build_controlled_manifests(
    records_by_source: dict[str, Sequence[BenchmarkExample]], *, output_root: str | Path,
    source_hashes: dict[str, str], config_hash: str, seed: int, samples_per_source: int,
) -> dict[str, ManifestHeader]:
    """Select and freeze all source manifests from already audited mappings."""
    headers: dict[str, ManifestHeader] = {}
    for source, records in records_by_source.items():
        dimensions = (
            ("risk_category", "threat_domain", "attack_type") if source == "jailbound"
            else ("risk_category", "attack_type")
        )
        selected, _ = select_controlled(
            records,
            n=samples_per_source,
            seed=seed,
            coverage_dimensions=dimensions,
        )
        headers[source] = write_controlled_manifest(output_root, source, selected, source_file_sha256=source_hashes[source], config_hash=config_hash)
    return headers


def _v2_payload_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _write_no_clobber(path: Path, payload: bytes, *, mismatch: str) -> None:
    """Create one immutable file, accepting only an exact-byte rerun."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(mismatch)
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(mismatch)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _locked_immutable_write(
    path: Path, payload: bytes, *, mismatch: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _write_no_clobber(path, payload, mismatch=mismatch)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _require_v2_records(records: Sequence[object]) -> None:
    if any(
        not isinstance(row, V2BenchmarkExample)
        or row.schema_version != "reviewer_eval.v2"
        for row in records
    ):
        raise ValueError("v2 manifests require V2BenchmarkExample records")


@dataclass(frozen=True)
class _PreparedV2Manifest:
    header: ManifestHeader
    manifest_path: Path
    header_path: Path
    manifest_bytes: bytes
    header_bytes: bytes


def _prepare_v2_controlled_manifest(
    output_root: str | Path,
    source: str,
    records: Sequence[V2BenchmarkExample],
    *,
    source_file_sha256: str,
    config_hash: str,
) -> _PreparedV2Manifest:
    if not source or Path(source).name != source:
        raise ValueError("manifest source must be a non-empty path-safe name")
    _require_v2_records(records)
    manifests = Path(output_root) / "manifests" / "v2"
    path = manifests / f"controlled_{source}.jsonl"
    header_path = manifests / f"controlled_{source}.header.json"
    ordered = tuple(sorted(records, key=lambda row: row.example_id))
    if any(row.source != source for row in ordered):
        raise ValueError("v2 manifest records must match the requested source")
    payloads = [row.model_dump(mode="json") for row in ordered]
    header = ManifestHeader(
        schema_version="reviewer_eval.v2",
        manifest_hash=canonical_hash(payloads),
        source=source,
        source_file_sha256=source_file_sha256,
        config_hash=config_hash,
        record_count=len(ordered),
        ordered_example_ids=tuple(row.example_id for row in ordered),
    )
    manifest_bytes = _v2_payload_bytes(payloads)
    header_bytes = (
        canonical_json(header.model_dump(mode="json")) + "\n"
    ).encode("utf-8")
    return _PreparedV2Manifest(
        header=header,
        manifest_path=path,
        header_path=header_path,
        manifest_bytes=manifest_bytes,
        header_bytes=header_bytes,
    )


def _validate_prepared_v2_manifest(
    prepared: _PreparedV2Manifest, *, mismatch: str
) -> None:
    for path, payload in (
        (prepared.manifest_path, prepared.manifest_bytes),
        (prepared.header_path, prepared.header_bytes),
    ):
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(mismatch)


def _commit_prepared_v2_manifests(
    prepared_manifests: Sequence[_PreparedV2Manifest],
) -> None:
    if not prepared_manifests:
        return
    manifests = prepared_manifests[0].manifest_path.parent
    if any(item.manifest_path.parent != manifests for item in prepared_manifests):
        raise ValueError("prepared v2 manifests must share one output root")
    manifests.mkdir(parents=True, exist_ok=True)
    lock_path = manifests / ".controlled_manifests.build.lock"
    mismatch = "immutable v2 manifest differs from existing output"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            for prepared in prepared_manifests:
                _validate_prepared_v2_manifest(prepared, mismatch=mismatch)
            for prepared in prepared_manifests:
                _write_no_clobber(
                    prepared.manifest_path,
                    prepared.manifest_bytes,
                    mismatch=mismatch,
                )
                _write_no_clobber(
                    prepared.header_path,
                    prepared.header_bytes,
                    mismatch=mismatch,
                )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def write_v2_controlled_manifest(
    output_root: str | Path,
    source: str,
    records: Sequence[V2BenchmarkExample],
    *,
    source_file_sha256: str,
    config_hash: str,
) -> ManifestHeader:
    """Freeze one schema-v2 manifest without overwriting concurrent output."""
    prepared = _prepare_v2_controlled_manifest(
        output_root,
        source,
        records,
        source_file_sha256=source_file_sha256,
        config_hash=config_hash,
    )
    _commit_prepared_v2_manifests((prepared,))
    return prepared.header


def write_v2_annotation_failures(
    output_root: str | Path, failures: Sequence[AnnotationFailure]
) -> Path:
    """Freeze the content-free annotation failure ledger."""
    path = Path(output_root) / "manifests" / "v2" / "annotation_failures.jsonl"
    ordered = sorted(
        failures,
        key=lambda row: (row.source, row.source_row, row.example_id),
    )
    _locked_immutable_write(
        path,
        _v2_payload_bytes([row.as_dict() for row in ordered]),
        mismatch="immutable v2 annotation failure ledger differs from existing output",
    )
    return path


def write_v2_build_report(
    output_root: str | Path, report: Mapping[str, object]
) -> Path:
    """Freeze source and failure counts next to the v2 manifests."""
    path = Path(output_root) / "manifests" / "v2" / "source_ingestion_report.json"
    payload = (canonical_json(dict(report)) + "\n").encode("utf-8")
    _locked_immutable_write(
        path,
        payload,
        mismatch="immutable v2 source ingestion report differs from existing output",
    )
    return path


def build_v2_controlled_manifests(
    records_by_source: dict[str, Sequence[V2BenchmarkExample]],
    *,
    output_root: str | Path,
    source_hashes: dict[str, str],
    config_hash: str,
    seed: int,
    samples_per_source: int,
) -> dict[str, ManifestHeader]:
    """Select only successfully annotated, deduplicated v2 candidates."""
    for records in records_by_source.values():
        _require_v2_records(records)

    prepared_manifests: list[_PreparedV2Manifest] = []
    for source, records in records_by_source.items():
        dimensions = (
            ("risk_category", "threat_domain", "attack_type")
            if source == "jailbound"
            else ("risk_category", "attack_type")
        )
        selected, _ = select_controlled(
            records,
            n=samples_per_source,
            seed=seed,
            coverage_dimensions=dimensions,
        )
        prepared_manifests.append(_prepare_v2_controlled_manifest(
            output_root,
            source,
            selected,
            source_file_sha256=source_hashes[source],
            config_hash=config_hash,
        ))
    _commit_prepared_v2_manifests(prepared_manifests)
    return {
        prepared.header.source: prepared.header for prepared in prepared_manifests
    }


def _annotation_failure(
    raw: RawExample,
    *,
    source_file: str,
    source_sha256: str,
    kind: FailureKind,
) -> AnnotationFailure:
    reason = {
        FailureKind.annotation: "annotation_error",
        FailureKind.transport: "transport_error",
    }[kind]
    return AnnotationFailure(
        schema_version="reviewer_eval.v2",
        example_id=raw.source_row_id,
        source=raw.source,
        source_file=source_file,
        source_row=raw.source_row,
        source_sha256=source_sha256,
        prompt_sha256=hashlib.sha256(raw.attack_text.encode("utf-8")).hexdigest(),
        intent_sha256=hashlib.sha256(raw.intent.encode("utf-8")).hexdigest(),
        failure_kind=kind,
        failure_reason=reason,
    )


def annotate_raw_candidates(
    records: Sequence[RawExample],
    *,
    annotator: CandidateAnnotator,
    taxonomy_mapper: Callable[[RawExample], MappingDecision],
    source_file: str,
    source_sha256: str,
    seed: int,
) -> tuple[tuple[V2BenchmarkExample, ...], tuple[AnnotationFailure, ...]]:
    """Annotate every raw candidate before it becomes selection-eligible."""
    converted: list[V2BenchmarkExample] = []
    failures: list[AnnotationFailure] = []
    for raw in records:
        source_hints: dict[str, object] = {
            "source": raw.source,
            "source_row": raw.source_row,
            "source_row_id": raw.source_row_id,
            "attack_label": raw.source_attack_label,
            "domain_label": raw.source_domain_label,
            "language": raw.language,
            "risk_label": raw.source_risk_label,
            "target_text": raw.target_text,
            "preprocessing": list(raw.preprocessing),
        }
        try:
            annotation = annotator.annotate(
                raw.attack_text,
                seed_intent=raw.intent,
                source_hints=source_hints,
            )
        except SpanAnnotationError:
            failures.append(
                _annotation_failure(
                    raw,
                    source_file=source_file,
                    source_sha256=source_sha256,
                    kind=FailureKind.annotation,
                )
            )
            continue
        except Exception:
            failures.append(
                _annotation_failure(
                    raw,
                    source_file=source_file,
                    source_sha256=source_sha256,
                    kind=FailureKind.transport,
                )
            )
            continue

        decision = taxonomy_mapper(raw)
        converted.append(
            V2BenchmarkExample(
                schema_version="reviewer_eval.v2",
                example_id=raw.source_row_id,
                source=raw.source,
                source_file=source_file,
                source_row=raw.source_row,
                source_sha256=source_sha256,
                intent=raw.intent,
                attack_text=raw.attack_text,
                target_text=raw.target_text,
                source_risk_label=raw.source_risk_label,
                source_attack_label=raw.source_attack_label,
                risk_category=decision.risk_category,
                threat_domain=decision.threat_domain,
                attack_type=decision.attack_type,
                language=raw.language,
                selection_stratum=(
                    f"{decision.risk_category}|{decision.attack_type}"
                ),
                selection_seed=seed,
                prompt_sha256=hashlib.sha256(
                    raw.attack_text.encode("utf-8")
                ).hexdigest(),
                preprocessing=(
                    raw.preprocessing
                    + decision.preprocessing
                    + ("model_annotated_editable_spans",)
                ),
                intent_sha256=hashlib.sha256(
                    raw.intent.encode("utf-8")
                ).hexdigest(),
                editable_spans=annotation.spans,
                annotator_model=annotation.model,
                annotator_revision=annotation.revision,
                annotation_template_sha256=annotation.template_sha256,
                annotation_response_sha256=annotation.response_sha256,
                annotation_confidence=annotation.confidence,
            )
        )
    return tuple(converted), tuple(failures)


def map_raw_candidates(
    records: Sequence[RawExample], *, mapping: dict, label_embeddings: dict[str, np.ndarray],
    encoder: Encoder, source_file: str, source_sha256: str, seed: int,
) -> tuple[BenchmarkExample, ...]:
    """Convert preselected raw rows to fully auditable manifest candidates."""
    converted: list[BenchmarkExample] = []
    for raw in records:
        decision = map_raw_example(raw, mapping, label_embeddings, encoder)
        prompt_hash = hashlib.sha256(raw.attack_text.encode("utf-8")).hexdigest()
        converted.append(BenchmarkExample(
            example_id=raw.source_row_id, source=raw.source, source_file=source_file,
            source_row=raw.source_row, source_sha256=source_sha256, intent=raw.intent,
            attack_text=raw.attack_text, target_text=raw.target_text,
            source_risk_label=raw.source_risk_label, source_attack_label=raw.source_attack_label,
            risk_category=decision.risk_category, threat_domain=decision.threat_domain,
            attack_type=decision.attack_type, language=raw.language,
            selection_stratum=f"{decision.risk_category}|{decision.attack_type}",
            selection_seed=seed, prompt_sha256=prompt_hash,
            preprocessing=raw.preprocessing + decision.preprocessing,
        ))
    return tuple(converted)


def _tie(seed: int, source: str, example_id: str) -> str:
    return hashlib.sha256(f"{seed}|{source}|{example_id}".encode()).hexdigest()


def select_controlled(
    records: Iterable[BenchmarkExample], *, n: int, seed: int,
    coverage_dimensions: Sequence[str],
) -> tuple[tuple[BenchmarkExample, ...], SelectionReport]:
    """Select a fixed-size, order-invariant, coverage-first subset."""
    record_list = tuple(records)
    example_ids = [record.example_id for record in record_list]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("controlled selection example_id values must be unique")
    by_prompt: dict[str, BenchmarkExample] = {}
    for record in sorted(record_list, key=lambda row: row.example_id):
        by_prompt.setdefault(record.prompt_sha256, record)
    eligible = sorted(by_prompt.values(), key=lambda row: row.example_id)
    if len(eligible) < n:
        raise ValueError(f"need {n} unique eligible prompts, found {len(eligible)}")

    frequency = {dimension: Counter(getattr(row, dimension) for row in eligible) for dimension in coverage_dimensions}
    uncovered = {dimension: set(frequency[dimension]) for dimension in coverage_dimensions}
    selected: list[BenchmarkExample] = []
    remaining = set(row.example_id for row in eligible)
    by_id = {row.example_id: row for row in eligible}
    while remaining and len(selected) < n and any(uncovered.values()):
        def rank(row: BenchmarkExample) -> tuple[float, float, str]:
            coverage = sum(getattr(row, dim) in uncovered[dim] for dim in coverage_dimensions)
            rarity = sum(1 / frequency[dim][getattr(row, dim)] for dim in coverage_dimensions)
            return (-coverage, -rarity, _tie(seed, row.source, row.example_id))
        chosen = min((by_id[item] for item in remaining), key=rank)
        selected.append(chosen); remaining.remove(chosen.example_id)
        for dimension in coverage_dimensions:
            uncovered[dimension].discard(getattr(chosen, dimension))

    strata = Counter(row.selection_stratum for row in eligible)
    quotas = {key: n * count / len(eligible) for key, count in strata.items()}
    targets = {key: int(value) for key, value in quotas.items()}
    for key, _ in sorted(quotas.items(), key=lambda pair: (-(pair[1] % 1), pair[0]))[: n - sum(targets.values())]:
        targets[key] += 1
    selected_by_stratum = Counter(row.selection_stratum for row in selected)
    candidates = sorted((by_id[item] for item in remaining), key=lambda row: _tie(seed, row.source, row.example_id))
    for row in candidates:
        if len(selected) == n: break
        if selected_by_stratum[row.selection_stratum] < targets[row.selection_stratum]:
            selected.append(row); selected_by_stratum[row.selection_stratum] += 1
    for row in candidates:
        if len(selected) == n: break
        if row not in selected:
            selected.append(row)
    if len(selected) != n:
        raise RuntimeError(
            f"controlled selection quota invariant failed: selected {len(selected)} of {n}"
        )
    selected.sort(key=lambda row: row.example_id)
    return tuple(selected), SelectionReport(len(record_list) - len(eligible), len(eligible), len(selected))


_FOL_MATCHING_CALIPERS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def select_fol_validation(rows: Sequence[FolCandidate], *, validation_n: int, low_n: int, middle_n: int, high_n: int) -> FolSplit:
    if validation_n != low_n + middle_n + high_n:
        raise ValueError("FOL split sizes must sum to validation_n")
    ordered = sorted(rows, key=lambda row: (row.fol, row.sample_id))
    if len(ordered) < validation_n:
        raise ValueError("insufficient FOL candidates")
    low_pool, high_pool = ordered[:18], ordered[-18:]
    values = np.array([[row.attack_loss, row.token_length, row.perplexity] for row in ordered], dtype=float)
    scale = np.maximum(values.std(axis=0), 1e-12)
    matched: tuple[float, list[tuple[FolCandidate, FolCandidate, float]]] | None = None
    for caliper in _FOL_MATCHING_CALIPERS:
        matrix = np.full((len(low_pool), len(high_pool)), 1e6)
        distances: dict[tuple[int, int], float] = {}
        for i, low in enumerate(low_pool):
            for j, high in enumerate(high_pool):
                if low.risk_category != high.risk_category or low.initial_label != high.initial_label:
                    continue
                distance = np.abs((np.array([low.attack_loss, low.token_length, low.perplexity]) - np.array([high.attack_loss, high.token_length, high.perplexity])) / scale)
                maximum = float(distance.max())
                if maximum <= caliper:
                    matrix[i, j] = float(distance.sum())
                    distances[(i, j)] = maximum
        left, right = linear_sum_assignment(matrix)
        pairs = [
            (low_pool[i], high_pool[j], distances[(i, j)])
            for i, j in zip(left, right)
            if (i, j) in distances
        ]
        if len(pairs) >= low_n:
            matched = (caliper, pairs[:low_n])
            break
    if matched is None:
        return FolSplit((), (), (), "inconclusive", tuple(row.sample_id for row in ordered))
    caliper, pairs = matched
    low, high = tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
    used = {row.sample_id for row in low + high}
    median = np.median([row.fol for row in ordered])
    middle = tuple(sorted((row for row in ordered if row.sample_id not in used), key=lambda row: (abs(row.fol - median), row.sample_id))[:middle_n])
    return FolSplit(
        low, middle, high, "ready",
        matching_caliper=caliper,
        matching_distances=tuple(pair[2] for pair in pairs),
    )
