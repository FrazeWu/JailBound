"""Deterministic sampling contracts for controlled reviewer experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io import atomic_write_json, canonical_hash, canonical_json
from .schema import BenchmarkExample, ManifestHeader
from .datasets import RawExample
from .semantic import Encoder, map_raw_example


@dataclass(frozen=True)
class SelectionReport:
    duplicate_count: int
    eligible_count: int
    selected_count: int


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
    selected.sort(key=lambda row: row.example_id)
    return tuple(selected), SelectionReport(len(record_list) - len(eligible), len(eligible), len(selected))


def select_fol_validation(rows: Sequence[FolCandidate], *, validation_n: int, low_n: int, middle_n: int, high_n: int) -> FolSplit:
    if validation_n != low_n + middle_n + high_n:
        raise ValueError("FOL split sizes must sum to validation_n")
    ordered = sorted(rows, key=lambda row: (row.fol, row.sample_id))
    if len(ordered) < validation_n:
        raise ValueError("insufficient FOL candidates")
    low_pool, high_pool = ordered[:18], ordered[-18:]
    matrix = np.full((len(low_pool), len(high_pool)), 1e6)
    values = np.array([[row.attack_loss, row.token_length, row.perplexity] for row in ordered], dtype=float)
    scale = np.maximum(values.std(axis=0), 1e-12)
    for i, low in enumerate(low_pool):
        for j, high in enumerate(high_pool):
            if low.risk_category != high.risk_category or low.initial_label != high.initial_label:
                continue
            distance = np.abs((np.array([low.attack_loss, low.token_length, low.perplexity]) - np.array([high.attack_loss, high.token_length, high.perplexity])) / scale)
            if np.all(distance <= .5): matrix[i, j] = float(distance.sum())
    left, right = linear_sum_assignment(matrix)
    pairs = [(low_pool[i], high_pool[j]) for i, j in zip(left, right) if matrix[i, j] < 1e6][:low_n]
    if len(pairs) < low_n:
        return FolSplit((), (), (), "inconclusive", tuple(row.sample_id for row in ordered))
    low, high = tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
    used = {row.sample_id for row in low + high}
    median = np.median([row.fol for row in ordered])
    middle = tuple(sorted((row for row in ordered if row.sample_id not in used), key=lambda row: (abs(row.fol - median), row.sample_id))[:middle_n])
    return FolSplit(low, middle, high, "ready")
