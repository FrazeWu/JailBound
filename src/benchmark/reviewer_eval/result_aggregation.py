"""Count-first aggregation of terminal judgment ledgers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

from .analysis import exact_mcnemar_pvalue
from .io import read_jsonl
from .metrics import Outcome, OutcomeSummary, summarize_outcomes


@dataclass(frozen=True)
class JudgmentSummaryRow:
    judge_key: str
    target_key: str
    source: str
    method: str
    threshold: float
    itt_asr: object
    execution_asr: object
    failed_count: int


@dataclass(frozen=True)
class PairedJudgmentDifference:
    """One method-versus-Init ITT comparison on a shared sample set."""

    judge_key: str
    target_key: str
    source: str
    method: str
    threshold: float
    denominator: int
    delta_itt: float
    method_only: int
    baseline_only: int
    mcnemar_pvalue: float


@dataclass(frozen=True)
class MaterializationSummaryRow:
    """Final-checkpoint materialization fidelity counts and numeric summaries."""

    source: str
    method: str
    total_count: int
    complete_count: int
    failed_count: int
    intent_preserved_count: int
    semantic_similarity_mean: float | None
    prefix_projection_cosine_mean: float | None
    seed_projection_cosine_mean: float | None


def summarize_judgments(rows: Iterable[Mapping[str, object]]) -> tuple[JudgmentSummaryRow, ...]:
    """Aggregate terminal judge rows into strict source/method/threshold cells."""
    grouped: dict[tuple[str, str, str, str, float], list[Outcome]] = {}
    for row in rows:
        try:
            judge_key = row["judge_key"]
            target_key = row["target_key"]
            source = row["source"]
            method = row["method"]
            threshold = row["threshold"]
            status = row["status"]
        except KeyError as error:
            raise ValueError(f"judgment row is missing {error.args[0]}") from error
        if not all(isinstance(value, str) and value for value in (judge_key, target_key, source, method)):
            raise ValueError("judgment grouping fields must be non-empty strings")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError("judgment threshold must be numeric")
        key = (judge_key, target_key, source, method, float(threshold))
        if status == "complete":
            unsafe = row.get("unsafe_label")
            if type(unsafe) is not bool:
                raise ValueError("complete judgment rows require a boolean unsafe_label")
            outcome = Outcome.complete(unsafe)
        elif status == "failed":
            outcome = Outcome.failed(str(row.get("failure_kind") or "judge"))
        else:
            raise ValueError("judgment rows must be terminal")
        grouped.setdefault(key, []).append(outcome)

    summaries = []
    for key, outcomes in sorted(grouped.items()):
        summary: OutcomeSummary = summarize_outcomes(outcomes)
        summaries.append(JudgmentSummaryRow(*key, summary.itt_asr, summary.execution_asr, summary.failed_count))
    return tuple(summaries)


def paired_judgment_differences(
    rows: Iterable[Mapping[str, object]],
) -> tuple[PairedJudgmentDifference, ...]:
    """Compare every terminal method cell against Init on identical IDs."""
    grouped: dict[tuple[str, str, str, float], dict[str, dict[str, bool]]] = {}
    for row in rows:
        try:
            judge_key = row["judge_key"]
            target_key = row["target_key"]
            source = row["source"]
            method = row["method"]
            sample_id = row["sample_id"]
            threshold = row["threshold"]
            status = row["status"]
        except KeyError as error:
            raise ValueError(f"judgment row is missing {error.args[0]}") from error
        if not all(isinstance(value, str) and value for value in (judge_key, target_key, source, method, sample_id)):
            raise ValueError("paired judgment fields must be non-empty strings")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("paired judgment threshold must be numeric")
        if status == "complete":
            unsafe = row.get("unsafe_label")
            if type(unsafe) is not bool:
                raise ValueError("complete judgment rows require a boolean unsafe_label")
            label = unsafe
        elif status == "failed":
            label = False
        else:
            raise ValueError("paired judgment rows must be terminal")
        methods = grouped.setdefault((judge_key, target_key, source, float(threshold)), {})
        sample_labels = methods.setdefault(method, {})
        if sample_id in sample_labels:
            raise ValueError("duplicate paired judgment sample")
        sample_labels[sample_id] = label

    comparisons: list[PairedJudgmentDifference] = []
    for (judge_key, target_key, source, threshold), methods in sorted(grouped.items()):
        baseline = methods.get("init")
        if baseline is None:
            raise ValueError("paired judgment comparison requires an init baseline")
        for method, labels in sorted(methods.items()):
            if method == "init":
                continue
            if set(labels) != set(baseline):
                raise ValueError("paired judgment comparison requires identical sample IDs")
            denominator = len(baseline)
            if denominator == 0:
                raise ValueError("paired judgment comparison requires at least one sample")
            method_only = sum(not baseline[sample_id] and labels[sample_id] for sample_id in baseline)
            baseline_only = sum(baseline[sample_id] and not labels[sample_id] for sample_id in baseline)
            delta = (sum(labels.values()) - sum(baseline.values())) / denominator
            comparisons.append(PairedJudgmentDifference(
                judge_key=judge_key,
                target_key=target_key,
                source=source,
                method=method,
                threshold=threshold,
                denominator=denominator,
                delta_itt=delta,
                method_only=method_only,
                baseline_only=baseline_only,
                mcnemar_pvalue=exact_mcnemar_pvalue(
                    method_only=method_only,
                    baseline_only=baseline_only,
                ),
            ))
    return tuple(comparisons)


def summarize_materializations(
    rows: Iterable[Mapping[str, object]],
) -> tuple[MaterializationSummaryRow, ...]:
    """Summarize only final materializations while retaining failures as counts."""
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        source, method = row.get("source"), row.get("method")
        checkpoint, status = row.get("checkpoint"), row.get("status")
        if not isinstance(source, str) or not source or not isinstance(method, str) or not method:
            raise ValueError("materialization source and method must be non-empty strings")
        if isinstance(checkpoint, bool) or not isinstance(checkpoint, int):
            raise ValueError("materialization checkpoint must be an integer")
        if status not in {"complete", "failed"}:
            raise ValueError("materialization rows must be terminal")
        final_checkpoint = 0 if method == "init" else 100
        if checkpoint == final_checkpoint:
            grouped.setdefault((source, method), []).append(row)

    summaries: list[MaterializationSummaryRow] = []
    for (source, method), records in sorted(grouped.items()):
        complete = [row for row in records if row["status"] == "complete"]
        semantic: list[float] = []
        prefix_cosine: list[float] = []
        seed_cosine: list[float] = []
        for row in complete:
            preserved = row.get("intent_preserved")
            if type(preserved) is not bool:
                raise ValueError("complete materializations require intent_preserved")
            for key, target in (
                ("semantic_similarity_after", semantic),
                ("prefix_projection_cosine", prefix_cosine),
                ("seed_projection_cosine", seed_cosine),
            ):
                value = row.get(key)
                if value is None and key != "semantic_similarity_after":
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"complete materialization has invalid {key}")
                target.append(float(value))
        summaries.append(MaterializationSummaryRow(
            source=source,
            method=method,
            total_count=len(records),
            complete_count=len(complete),
            failed_count=len(records) - len(complete),
            intent_preserved_count=sum(row["intent_preserved"] is True for row in complete),
            semantic_similarity_mean=(sum(semantic) / len(semantic)) if semantic else None,
            prefix_projection_cosine_mean=(sum(prefix_cosine) / len(prefix_cosine)) if prefix_cosine else None,
            seed_projection_cosine_mean=(sum(seed_cosine) / len(seed_cosine)) if seed_cosine else None,
        ))
    return tuple(summaries)


def load_judgment_rows(output_root: str | Path) -> list[dict[str, object]]:
    """Read every persisted judge ledger below one immutable experiment root."""
    root = Path(output_root) / "judgments"
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/*/*/*/records.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def write_paired_judgment_differences(output_root: str | Path) -> Path:
    """Write method-versus-Init paired ITT comparisons for every judge cell."""
    root = Path(output_root)
    destination = root / "analysis" / "paired_asr.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = paired_judgment_differences(load_judgment_rows(root))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Judge", "Target", "Source", "Method", "Threshold", "Denominator",
                "Net ASR change", "Net ASR change (%)", "Method-only", "Init-only",
                "McNemar p", "Provenance",
            ),
        )
        writer.writeheader()
        for row in rows:
            net_count = row.method_only - row.baseline_only
            writer.writerow({
                "Judge": row.judge_key,
                "Target": row.target_key,
                "Source": row.source,
                "Method": row.method,
                "Threshold": row.threshold,
                "Denominator": row.denominator,
                "Net ASR change": f"{net_count} / {row.denominator}",
                "Net ASR change (%)": f"{row.delta_itt * 100:.2f}%",
                "Method-only": row.method_only,
                "Init-only": row.baseline_only,
                "McNemar p": f"{row.mcnemar_pvalue:.6g}",
                "Provenance": "new_run",
            })
    return destination


def load_materialization_rows(output_root: str | Path) -> list[dict[str, object]]:
    """Read all persisted materialization ledgers under one experiment root."""
    root = Path(output_root) / "optimization"
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/*/materialization.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def write_materialization_summaries(output_root: str | Path) -> Path:
    """Write final materialization fidelity summaries for review-facing tables."""
    root = Path(output_root)
    destination = root / "analysis" / "materialization_fidelity.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize_materializations(load_materialization_rows(root))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Source", "Method", "Total", "Complete", "Failed", "Intent preserved",
                "Semantic similarity mean", "Prefix projection cosine mean",
                "Seed projection cosine mean", "Provenance",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Source": row.source,
                "Method": row.method,
                "Total": row.total_count,
                "Complete": row.complete_count,
                "Failed": row.failed_count,
                "Intent preserved": f"{row.intent_preserved_count} / {row.total_count}",
                "Semantic similarity mean": row.semantic_similarity_mean,
                "Prefix projection cosine mean": row.prefix_projection_cosine_mean,
                "Seed projection cosine mean": row.seed_projection_cosine_mean,
                "Provenance": "new_run",
            })
    return destination


def write_judgment_summaries(output_root: str | Path) -> Path:
    """Write the review-facing judge sensitivity table with count-first rates."""
    root = Path(output_root)
    destination = root / "analysis" / "judge_sensitivity.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize_judgments(load_judgment_rows(root))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Judge", "Target", "Source", "Method", "Threshold",
                "ITT ASR", "Execution ASR", "Failed", "Provenance",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Judge": row.judge_key,
                "Target": row.target_key,
                "Source": row.source,
                "Method": row.method,
                "Threshold": row.threshold,
                "ITT ASR": row.itt_asr.display,
                "Execution ASR": row.execution_asr.display,
                "Failed": row.failed_count,
                "Provenance": "new_run",
            })
    return destination
