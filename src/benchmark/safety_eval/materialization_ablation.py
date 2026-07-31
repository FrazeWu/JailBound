"""Strict records and paired summaries for the materialization ablation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, StringConstraints, model_validator


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class Branch(StrEnum):
    """Canonical reviewer-facing optimization branches."""

    high_value = "High-Value"
    safety_sensitivity = "Safety-Sensitivity"


class BinaryPairJudgment(BaseModel):
    """Unsafe labels for both conditions from one frozen judge."""

    model_config = ConfigDict(extra="forbid")

    continuous: StrictBool
    materialized: StrictBool


class MaterializationPair(BaseModel):
    """One same-state continuous/materialized comparison."""

    model_config = ConfigDict(extra="forbid")

    source: StrictStr
    sample_id: StrictStr
    branch: Branch
    optimization_checkpoint: StrictInt
    state_sha256: Sha256 | None
    model_key: StrictStr
    model_revision: StrictStr
    initial_discrete_prompt: StrictStr
    reference_intent: StrictStr
    continuous_response: StrictStr
    materialized_text: StrictStr
    materialized_response: StrictStr
    editable_projected_token_ids: tuple[StrictInt, ...]
    projected_token_ids: tuple[StrictInt, ...]
    retokenized_token_ids: tuple[StrictInt, ...]
    projection_cosines: tuple[float, ...]
    roundtrip_exact_match: StrictBool
    projected_length: StrictInt
    retokenized_length: StrictInt
    max_new_tokens: StrictInt
    status: Literal["complete", "failed"]
    error: StrictStr | None
    judgments: dict[str, BinaryPairJudgment]

    @model_validator(mode="after")
    def validate_evidence(self) -> "MaterializationPair":
        for field_name in ("source", "sample_id", "model_key", "model_revision"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.optimization_checkpoint < 0:
            raise ValueError("optimization checkpoint must be non-negative")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.projected_length != len(self.projected_token_ids):
            raise ValueError("projected length does not match token IDs")
        if self.retokenized_length != len(self.retokenized_token_ids):
            raise ValueError("retokenized length does not match token IDs")
        if len(self.projection_cosines) != len(self.editable_projected_token_ids):
            raise ValueError("projection cosine count does not match editable projected token IDs")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in self.projection_cosines):
            raise ValueError("projection cosines must be finite and in [-1, 1]")
        if self.status == "failed":
            if not self.error:
                raise ValueError("failed pair requires an error")
        else:
            if self.state_sha256 is None:
                raise ValueError("complete pair requires a state hash")
            exact = self.projected_token_ids == self.retokenized_token_ids
            if self.roundtrip_exact_match != exact:
                raise ValueError("round-trip flag disagrees with token IDs")
            if self.error is not None:
                raise ValueError("complete pair cannot contain an error")
            required_text = (
                self.initial_discrete_prompt,
                self.reference_intent,
                self.continuous_response,
                self.materialized_text,
                self.materialized_response,
            )
            if any(not value.strip() for value in required_text):
                raise ValueError("complete pair requires all text evidence")
            if not self.projected_token_ids or not self.editable_projected_token_ids:
                raise ValueError("complete pair requires projection evidence")
        return self


PairKey = tuple[str, str, Branch, int]


def canonical_pair_key(pair: MaterializationPair) -> PairKey:
    """Return the unique key required by the paired experiment."""
    return pair.source, pair.sample_id, pair.branch, pair.optimization_checkpoint


def index_pairs(pairs: Iterable[MaterializationPair]) -> dict[PairKey, MaterializationPair]:
    """Index pairs while rejecting ambiguous experiment units."""
    indexed: dict[PairKey, MaterializationPair] = {}
    for pair in pairs:
        key = canonical_pair_key(pair)
        if key in indexed:
            raise ValueError(f"duplicate materialization pair: {key}")
        indexed[key] = pair
    return indexed


@dataclass(frozen=True)
class PairSummary:
    judge_key: str
    source: str
    branch: Branch | Literal["All"]
    total_pairs: int
    valid_pairs: int
    failed_pairs: int
    continuous_unsafe: int
    materialized_unsafe: int
    retained_unsafe: int
    agreements: int
    continuous_asr: float | None
    materialized_asr: float | None
    difference_pp: float | None
    unsafe_retention: float | None
    label_agreement: float | None


@dataclass(frozen=True)
class RoundTripSummary:
    source: str
    branch: Branch | Literal["All"]
    total_pairs: int
    valid_pairs: int
    failed_pairs: int
    exact_roundtrips: int
    exact_roundtrip_rate: float | None
    mean_projected_length: float | None
    mean_retokenized_length: float | None
    mean_projection_cosine: float | None


def _groups(
    pairs: Iterable[MaterializationPair],
) -> dict[tuple[str, Branch | Literal["All"]], list[MaterializationPair]]:
    indexed = list(index_pairs(pairs).values())
    cross_groups: dict[tuple[str, Branch], list[MaterializationPair]] = defaultdict(list)
    for pair in indexed:
        cross_groups[(pair.source, pair.branch)].append(pair)
    grouped: dict[tuple[str, Branch | Literal["All"]], list[MaterializationPair]] = {}
    for key in sorted(cross_groups, key=lambda value: (value[0], value[1].value)):
        grouped[key] = cross_groups[key]
    grouped[("Overall", "All")] = indexed
    return grouped


def summarize_pairs(pairs: Iterable[MaterializationPair], *, judge_key: str) -> tuple[PairSummary, ...]:
    """Compute valid-pair-only behavioral fidelity metrics."""
    if not judge_key.strip():
        raise ValueError("judge key must be non-empty")
    summaries: list[PairSummary] = []
    for (source, branch), rows in _groups(pairs).items():
        valid = [row for row in rows if row.status == "complete"]
        missing = [row.sample_id for row in valid if judge_key not in row.judgments]
        if missing:
            raise ValueError(f"missing judge labels for {judge_key}: {sorted(missing)}")
        labels = [row.judgments[judge_key] for row in valid]
        denominator = len(labels)
        continuous_unsafe = sum(label.continuous for label in labels)
        materialized_unsafe = sum(label.materialized for label in labels)
        retained = sum(label.continuous and label.materialized for label in labels)
        agreements = sum(label.continuous == label.materialized for label in labels)
        continuous_asr = continuous_unsafe / denominator if denominator else None
        materialized_asr = materialized_unsafe / denominator if denominator else None
        summaries.append(PairSummary(
            judge_key=judge_key,
            source=source,
            branch=branch,
            total_pairs=len(rows),
            valid_pairs=denominator,
            failed_pairs=len(rows) - denominator,
            continuous_unsafe=continuous_unsafe,
            materialized_unsafe=materialized_unsafe,
            retained_unsafe=retained,
            agreements=agreements,
            continuous_asr=continuous_asr,
            materialized_asr=materialized_asr,
            difference_pp=(100.0 * (materialized_asr - continuous_asr)) if denominator else None,
            unsafe_retention=(retained / continuous_unsafe) if continuous_unsafe else None,
            label_agreement=(agreements / denominator) if denominator else None,
        ))
    return tuple(summaries)


def summarize_roundtrips(pairs: Iterable[MaterializationPair]) -> tuple[RoundTripSummary, ...]:
    """Summarize decode/re-tokenize evidence without filtering mismatches."""
    summaries: list[RoundTripSummary] = []
    for (source, branch), rows in _groups(pairs).items():
        valid = [row for row in rows if row.status == "complete"]
        cosines = [value for row in valid for value in row.projection_cosines]
        summaries.append(RoundTripSummary(
            source=source,
            branch=branch,
            total_pairs=len(rows),
            valid_pairs=len(valid),
            failed_pairs=len(rows) - len(valid),
            exact_roundtrips=sum(row.roundtrip_exact_match for row in valid),
            exact_roundtrip_rate=(sum(row.roundtrip_exact_match for row in valid) / len(valid)) if valid else None,
            mean_projected_length=(sum(row.projected_length for row in valid) / len(valid)) if valid else None,
            mean_retokenized_length=(sum(row.retokenized_length for row in valid) / len(valid)) if valid else None,
            mean_projection_cosine=(sum(cosines) / len(cosines)) if cosines else None,
        ))
    return tuple(summaries)


def _load_pairs(path: Path) -> tuple[MaterializationPair, ...]:
    rows: list[MaterializationPair] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(MaterializationPair.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid materialization pair at line {line_number}") from error
    index_pairs(rows)
    return tuple(rows)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pair_summaries(
    pairs_path: Path,
    output_dir: Path,
    *,
    judge_keys: tuple[str, ...],
) -> tuple[Path, Path, Path]:
    """Write count-first paired, round-trip, and failure artifacts."""
    if not judge_keys or any(not key.strip() for key in judge_keys):
        raise ValueError("at least one non-empty judge key is required")
    pairs = _load_pairs(pairs_path)
    pair_rows = [asdict(row) for key in judge_keys for row in summarize_pairs(pairs, judge_key=key)]
    roundtrip_rows = [asdict(row) for row in summarize_roundtrips(pairs)]
    for row in pair_rows:
        row["branch"] = row["branch"].value if isinstance(row["branch"], Branch) else row["branch"]
    for row in roundtrip_rows:
        row["branch"] = row["branch"].value if isinstance(row["branch"], Branch) else row["branch"]

    pair_path = output_dir / "materialization_ablation.csv"
    roundtrip_path = output_dir / "materialization_roundtrip.csv"
    failures_path = output_dir / "materialization_failures.json"
    pair_fields = (
        "Judge", "Source", "Branch", "Total pairs", "Valid pairs", "Failed pairs",
        "Continuous unsafe", "Materialized unsafe", "Retained unsafe", "Agreements",
        "Continuous ASR", "Materialized ASR", "Difference (pp)", "Unsafe retention", "Label agreement",
    )
    _write_csv(pair_path, pair_fields, ({
        "Judge": row["judge_key"], "Source": row["source"], "Branch": row["branch"],
        "Total pairs": row["total_pairs"], "Valid pairs": row["valid_pairs"],
        "Failed pairs": row["failed_pairs"], "Continuous unsafe": row["continuous_unsafe"],
        "Materialized unsafe": row["materialized_unsafe"], "Retained unsafe": row["retained_unsafe"],
        "Agreements": row["agreements"], "Continuous ASR": row["continuous_asr"],
        "Materialized ASR": row["materialized_asr"], "Difference (pp)": row["difference_pp"],
        "Unsafe retention": row["unsafe_retention"], "Label agreement": row["label_agreement"],
    } for row in pair_rows))
    roundtrip_fields = (
        "Source", "Branch", "Total pairs", "Valid pairs", "Failed pairs", "Exact round trips",
        "Exact round-trip rate", "Mean projected length", "Mean retokenized length", "Mean projection cosine",
    )
    _write_csv(roundtrip_path, roundtrip_fields, ({
        "Source": row["source"], "Branch": row["branch"], "Total pairs": row["total_pairs"],
        "Valid pairs": row["valid_pairs"], "Failed pairs": row["failed_pairs"],
        "Exact round trips": row["exact_roundtrips"], "Exact round-trip rate": row["exact_roundtrip_rate"],
        "Mean projected length": row["mean_projected_length"],
        "Mean retokenized length": row["mean_retokenized_length"],
        "Mean projection cosine": row["mean_projection_cosine"],
    } for row in roundtrip_rows))
    failures = [
        {
            "source": pair.source,
            "sample_id": pair.sample_id,
            "branch": pair.branch.value,
            "optimization_checkpoint": pair.optimization_checkpoint,
            "error": pair.error,
        }
        for pair in pairs
        if pair.status == "failed"
    ]
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pair_path, roundtrip_path, failures_path
