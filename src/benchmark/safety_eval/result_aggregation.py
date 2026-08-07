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
    schema_version: str = "reviewer_eval.v1"
    branch: str | None = None
    transport: str | None = None
    judge_revision: str | None = None
    target_revision: str | None = None
    target_tokenizer_sha256: str | None = None
    materialization_sha256s: tuple[str, ...] | None = None


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
    schema_version: str = "reviewer_eval.v1"
    branch: str | None = None
    transport: str | None = None
    judge_revision: str | None = None
    target_revision: str | None = None
    target_tokenizer_sha256: str | None = None
    materialization_sha256s: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MaterializationSummaryRow:
    """Final-checkpoint materialization fidelity counts and numeric summaries."""

    source: str
    method: str
    schema_version: str
    total_count: int
    complete_count: int
    failed_count: int
    intent_preserved_count: int
    semantic_similarity_mean: float | None
    prefix_projection_cosine_mean: float | None
    seed_projection_cosine_mean: float | None
    frozen_positions_unchanged_count: int = 0
    full_prompt_similarity_mean: float | None = None
    editable_span_similarity_mean: float | None = None


def summarize_judgments(rows: Iterable[Mapping[str, object]]) -> tuple[JudgmentSummaryRow, ...]:
    """Aggregate terminal judge rows into strict source/method/threshold cells."""
    grouped: dict[tuple[str, str, str, str, float, str, str | None, str | None, str | None, str | None, str | None], list[tuple[Outcome, str | None]]] = {}
    v2_samples: dict[tuple[str, str, str, str, float, str, str | None, str | None, str | None, str | None, str | None], set[str]] = {}
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
        schema_version = row.get("schema_version", "reviewer_eval.v1")
        branch = row.get("branch")
        transport = row.get("transport")
        if schema_version == "reviewer_eval.v2":
            if not isinstance(branch, str) or not branch or transport not in {"text", "embedding"}:
                raise ValueError("v2 judgment rows require branch and transport")
            target_revision = row.get("target_revision")
            target_tokenizer_sha256 = row.get("target_tokenizer_sha256")
            materialization_sha256 = row.get("materialization_sha256")
            judge_revision = row.get("judge_revision")
            sample_id = row.get("sample_id")
            if not all(isinstance(value, str) and value for value in (
                judge_revision, target_revision, target_tokenizer_sha256, materialization_sha256, sample_id,
            )):
                raise ValueError("v2 judgment rows require execution provenance")
        elif schema_version != "reviewer_eval.v1":
            raise ValueError("judgment rows require a supported schema version")
        else:
            branch = transport = None
            judge_revision = target_revision = target_tokenizer_sha256 = None
        key = (judge_key, target_key, source, method, float(threshold), schema_version, branch, transport, judge_revision, target_revision, target_tokenizer_sha256)
        if schema_version == "reviewer_eval.v2":
            samples = v2_samples.setdefault(key, set())
            if sample_id in samples:
                raise ValueError("duplicate v2 judgment sample for one target identity")
            samples.add(sample_id)
        if status == "complete":
            unsafe = row.get("unsafe_label")
            if type(unsafe) is not bool:
                raise ValueError("complete judgment rows require a boolean unsafe_label")
            outcome = Outcome.complete(unsafe)
        elif status == "failed":
            outcome = Outcome.failed(str(row.get("failure_kind") or "judge"))
        else:
            raise ValueError("judgment rows must be terminal")
        grouped.setdefault(key, []).append((outcome, materialization_sha256 if schema_version == "reviewer_eval.v2" else None))

    summaries = []
    for key, records in sorted(grouped.items()):
        summary: OutcomeSummary = summarize_outcomes(outcome for outcome, _ in records)
        materialization_sha256s = (
            tuple(sorted({value for _, value in records if value is not None}))
            if key[5] == "reviewer_eval.v2" else None
        )
        summaries.append(JudgmentSummaryRow(
            judge_key=key[0], target_key=key[1], source=key[2], method=key[3], threshold=key[4],
            itt_asr=summary.itt_asr, execution_asr=summary.execution_asr, failed_count=summary.failed_count,
            schema_version=key[5], branch=key[6], transport=key[7],
            judge_revision=key[8], target_revision=key[9], target_tokenizer_sha256=key[10],
            materialization_sha256s=materialization_sha256s,
        ))
    return tuple(summaries)


def paired_judgment_differences(
    rows: Iterable[Mapping[str, object]],
) -> tuple[PairedJudgmentDifference, ...]:
    """Compare every terminal method cell against Init on identical IDs."""
    grouped: dict[tuple[str, str, str, float, str, str | None, str | None, str | None, str | None, str | None], dict[str, dict[str, tuple[bool, str | None]]]] = {}
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
        schema_version = row.get("schema_version", "reviewer_eval.v1")
        if schema_version == "reviewer_eval.v2":
            branch = row.get("branch")
            transport = row.get("transport")
            target_revision = row.get("target_revision")
            target_tokenizer_sha256 = row.get("target_tokenizer_sha256")
            materialization_sha256 = row.get("materialization_sha256")
            judge_revision = row.get("judge_revision")
            if (
                not isinstance(branch, str) or not branch or transport != "text"
                or not all(isinstance(value, str) and value for value in (
                    judge_revision, target_revision, target_tokenizer_sha256, materialization_sha256,
                ))
            ):
                raise ValueError("v2 paired judgments require execution provenance")
        elif schema_version == "reviewer_eval.v1":
            branch = transport = judge_revision = target_revision = target_tokenizer_sha256 = None
        else:
            raise ValueError("paired judgment rows require a supported schema version")
        if status == "complete":
            unsafe = row.get("unsafe_label")
            if type(unsafe) is not bool:
                raise ValueError("complete judgment rows require a boolean unsafe_label")
            label = unsafe
        elif status == "failed":
            label = False
        else:
            raise ValueError("paired judgment rows must be terminal")
        methods = grouped.setdefault(
            (judge_key, target_key, source, float(threshold), schema_version, branch, transport, judge_revision, target_revision, target_tokenizer_sha256),
            {},
        )
        sample_outcomes = methods.setdefault(method, {})
        if sample_id in sample_outcomes:
            raise ValueError("duplicate paired judgment sample")
        sample_outcomes[sample_id] = (label, materialization_sha256 if schema_version == "reviewer_eval.v2" else None)

    comparisons: list[PairedJudgmentDifference] = []
    for (judge_key, target_key, source, threshold, _schema_version, _branch, _transport, _judge_revision, _target_revision, _target_tokenizer_sha256), methods in sorted(grouped.items()):
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
            method_only = sum(not baseline[sample_id][0] and labels[sample_id][0] for sample_id in baseline)
            baseline_only = sum(baseline[sample_id][0] and not labels[sample_id][0] for sample_id in baseline)
            delta = (sum(label for label, _ in labels.values()) - sum(label for label, _ in baseline.values())) / denominator
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
                schema_version=_schema_version,
                branch=_branch,
                transport=_transport,
                judge_revision=_judge_revision,
                target_revision=_target_revision,
                target_tokenizer_sha256=_target_tokenizer_sha256,
                materialization_sha256s=(
                    tuple(sorted(({
                        value for _, value in baseline.values()
                    } | {
                        value for _, value in labels.values()
                    }) - {None}))
                    if _schema_version == "reviewer_eval.v2" else None
                ),
            ))
    return tuple(comparisons)


def summarize_materializations(
    rows: Iterable[Mapping[str, object]],
) -> tuple[MaterializationSummaryRow, ...]:
    """Summarize only final materializations while retaining failures as counts."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        source, method = row.get("source"), row.get("method")
        schema_version = row.get("schema_version", "reviewer_eval.v1")
        checkpoint = row.get("step") if schema_version == "reviewer_eval.v2" else row.get("checkpoint")
        status = row.get("status")
        if not isinstance(source, str) or not source or not isinstance(method, str) or not method:
            raise ValueError("materialization source and method must be non-empty strings")
        if schema_version not in {"reviewer_eval.v1", "reviewer_eval.v2"}:
            raise ValueError("materialization rows require a supported schema version")
        if isinstance(checkpoint, bool) or not isinstance(checkpoint, int):
            raise ValueError("materialization checkpoint or step must be an integer")
        if status not in {"complete", "failed"}:
            raise ValueError("materialization rows must be terminal")
        final_checkpoint = 0 if method == "init" else 100
        if checkpoint == final_checkpoint:
            grouped.setdefault((source, method, schema_version), []).append(row)

    summaries: list[MaterializationSummaryRow] = []
    for (source, method, schema_version), records in sorted(grouped.items()):
        complete = [row for row in records if row["status"] == "complete"]
        if schema_version == "reviewer_eval.v2":
            frozen = 0
            full_similarity: list[float] = []
            editable_similarity: list[float] = []
            for row in complete:
                unchanged = row.get("frozen_positions_unchanged")
                if type(unchanged) is not bool:
                    raise ValueError("complete v2 materializations require frozen_positions_unchanged")
                frozen += int(unchanged)
                for key, target in (
                    ("full_prompt_similarity", full_similarity),
                    ("editable_span_similarity", editable_similarity),
                ):
                    value = row.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise ValueError(f"complete v2 materialization has invalid {key}")
                    target.append(float(value))
            summaries.append(MaterializationSummaryRow(
                source=source,
                method=method,
                schema_version=schema_version,
                total_count=len(records),
                complete_count=len(complete),
                failed_count=len(records) - len(complete),
                intent_preserved_count=0,
                semantic_similarity_mean=None,
                prefix_projection_cosine_mean=None,
                seed_projection_cosine_mean=None,
                frozen_positions_unchanged_count=frozen,
                full_prompt_similarity_mean=(sum(full_similarity) / len(full_similarity)) if full_similarity else None,
                editable_span_similarity_mean=(sum(editable_similarity) / len(editable_similarity)) if editable_similarity else None,
            ))
            continue
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
            source=source, method=method, schema_version=schema_version,
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
                "McNemar p", "Schema", "Branch", "Transport", "Judge revision", "Target revision",
                "Target tokenizer SHA256", "Materialization SHA256s", "Provenance",
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
                "Schema": row.schema_version,
                "Branch": row.branch,
                "Transport": row.transport,
                "Judge revision": row.judge_revision,
                "Target revision": row.target_revision,
                "Target tokenizer SHA256": row.target_tokenizer_sha256,
                "Materialization SHA256s": ",".join(row.materialization_sha256s or ()),
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
                "Source", "Method", "Schema", "Total", "Complete", "Failed", "Intent preserved",
                "Semantic similarity mean", "Prefix projection cosine mean",
                "Seed projection cosine mean", "Frozen positions unchanged",
                "Full prompt similarity mean", "Editable span similarity mean", "Provenance",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Source": row.source,
                "Method": row.method,
                "Schema": row.schema_version,
                "Total": row.total_count,
                "Complete": row.complete_count,
                "Failed": row.failed_count,
                "Intent preserved": f"{row.intent_preserved_count} / {row.total_count}",
                "Semantic similarity mean": row.semantic_similarity_mean,
                "Prefix projection cosine mean": row.prefix_projection_cosine_mean,
                "Seed projection cosine mean": row.seed_projection_cosine_mean,
                "Frozen positions unchanged": f"{row.frozen_positions_unchanged_count} / {row.total_count}",
                "Full prompt similarity mean": row.full_prompt_similarity_mean,
                "Editable span similarity mean": row.editable_span_similarity_mean,
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
                "ITT ASR", "Execution ASR", "Failed", "Schema", "Branch", "Transport", "Judge revision",
                "Target revision", "Target tokenizer SHA256", "Materialization SHA256s", "Provenance",
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
                "Schema": row.schema_version,
                "Branch": row.branch,
                "Transport": row.transport,
                "Judge revision": row.judge_revision,
                "Target revision": row.target_revision,
                "Target tokenizer SHA256": row.target_tokenizer_sha256,
                "Materialization SHA256s": ",".join(row.materialization_sha256s or ()),
                "Provenance": "new_run",
            })
    return destination
