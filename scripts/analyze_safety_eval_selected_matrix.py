"""Summarize only method-gated safety-evaluation results without exposing prompt text."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchmark.safety_eval.analysis import exact_mcnemar_pvalue, threshold_rank_sensitivity
from benchmark.safety_eval.io import atomic_write_json, read_jsonl
from benchmark.safety_eval.metrics import Rate


SECONDARY_JUDGE = "qwen32_compat"


def _read_manifest(selection_root: Path) -> tuple[tuple[str, ...], str, int]:
    payload = json.loads((selection_root / "selection_manifest.json").read_text(encoding="utf-8"))
    sources = tuple(payload.get("sources", ()))
    target = payload.get("target_key")
    sample_count = payload.get("samples_per_source")
    if not sources or len(sources) != len(set(sources)) or not all(isinstance(source, str) and source for source in sources):
        raise ValueError("selection manifest sources are invalid")
    if not isinstance(target, str) or not target:
        raise ValueError("selection manifest target is invalid")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("selection manifest sample count is invalid")
    return sources, target, sample_count


def _complete_methods(selection_root: Path, *, sources: tuple[str, ...], samples_per_source: int) -> tuple[str, ...]:
    methods: list[str] = []
    for path in sorted((selection_root / "methods").glob("*.complete.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = payload.get("method")
        expected_responses = len(sources) * samples_per_source
        if (
            not isinstance(method, str)
            or not method
            or payload.get("source_count") != len(sources)
            or payload.get("response_count") != expected_responses
            or not isinstance(payload.get("primary_judgment_count"), int)
            or not isinstance(payload.get("secondary_judgment_count"), int)
            or payload["primary_judgment_count"] != payload["secondary_judgment_count"]
        ):
            raise ValueError(f"invalid complete marker: {path.name}")
        methods.append(method)
    if not methods:
        raise ValueError("no completed method markers")
    if len(methods) != len(set(methods)):
        raise ValueError("duplicate completed method markers")
    if "init" not in methods:
        raise ValueError("completed methods require init")
    return tuple(methods)


def _terminal_labels(path: Path, *, checkpoint: int, sample_count: int) -> dict[float, dict[str, bool]]:
    rows = read_jsonl(path)
    values: dict[float, dict[str, bool]] = {}
    for row in rows:
        if row.get("checkpoint") != checkpoint:
            continue
        if row.get("status") != "complete" or type(row.get("unsafe_label")) is not bool:
            raise ValueError(f"non-complete judgment in {path}")
        sample_id, threshold = row.get("sample_id"), row.get("threshold")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"invalid sample identifier in {path}")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"invalid threshold in {path}")
        bucket = values.setdefault(float(threshold), {})
        if sample_id in bucket:
            raise ValueError(f"duplicate terminal judgment in {path}")
        bucket[sample_id] = row["unsafe_label"]
    if not values or any(len(labels) != sample_count for labels in values.values()):
        raise ValueError(f"incomplete terminal judgments in {path}")
    return values


def _labels_for_cell(
    *,
    parent_root: Path,
    selection_root: Path,
    target: str,
    source: str,
    method: str,
    samples_per_source: int,
    primary_judge: str,
) -> dict[str, dict[float, dict[str, bool]]]:
    checkpoint = 0 if method == "init" else 100
    primary = _terminal_labels(
        parent_root / "judgments" / primary_judge / target / source / method / "records.jsonl",
        checkpoint=checkpoint,
        sample_count=samples_per_source,
    )
    secondary = _terminal_labels(
        selection_root / "judgments" / SECONDARY_JUDGE / target / source / method / "records.jsonl",
        checkpoint=checkpoint,
        sample_count=samples_per_source,
    )
    if set(primary) != set(secondary):
        raise ValueError(f"judge thresholds disagree for {source}/{method}")
    for threshold in primary:
        if set(primary[threshold]) != set(secondary[threshold]):
            raise ValueError(f"judge sample IDs disagree for {source}/{method}")
    return {primary_judge: primary, SECONDARY_JUDGE: secondary}


def analyze_selected_matrix(
    *, parent_root: Path, selection_root: Path, primary_judge: str = "octopus_seval_14b"
) -> dict[str, Any]:
    """Write count-first tables from complete selected-matrix cells only."""
    if not primary_judge:
        raise ValueError("primary judge key is required")
    sources, target, samples_per_source = _read_manifest(selection_root)
    methods = _complete_methods(selection_root, sources=sources, samples_per_source=samples_per_source)
    labels: dict[tuple[str, str, str], dict[float, dict[str, bool]]] = {}
    for source in sources:
        for method in methods:
            for judge, by_threshold in _labels_for_cell(
                parent_root=parent_root,
                selection_root=selection_root,
                target=target,
                source=source,
                method=method,
                samples_per_source=samples_per_source,
                primary_judge=primary_judge,
            ).items():
                labels[(judge, source, method)] = by_threshold

    analysis_root = selection_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Judge", "Target", "Source", "Method", "Threshold", "Unsafe", "N", "ASR"))
        writer.writeheader()
        for (judge, source, method), by_threshold in sorted(labels.items()):
            for threshold, sample_labels in sorted(by_threshold.items()):
                rate = Rate.from_flags(sample_labels.values())
                writer.writerow({"Judge": judge, "Target": target, "Source": source, "Method": method, "Threshold": threshold, "Unsafe": rate.numerator, "N": rate.denominator, "ASR": rate.display})

    paired_path = analysis_root / "paired_vs_init.csv"
    with paired_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Judge", "Target", "Source", "Method", "Threshold", "N", "Net ASR change", "Method-only", "Init-only", "McNemar p"))
        writer.writeheader()
        for judge in (primary_judge, SECONDARY_JUDGE):
            for source in sources:
                baseline = labels[(judge, source, "init")]
                for method in methods:
                    if method == "init":
                        continue
                    for threshold, baseline_labels in sorted(baseline.items()):
                        method_labels = labels[(judge, source, method)][threshold]
                        method_only = sum(not baseline_labels[key] and method_labels[key] for key in baseline_labels)
                        init_only = sum(baseline_labels[key] and not method_labels[key] for key in baseline_labels)
                        writer.writerow({
                            "Judge": judge, "Target": target, "Source": source, "Method": method,
                            "Threshold": threshold, "N": len(baseline_labels),
                            "Net ASR change": f"{method_only - init_only} / {len(baseline_labels)}",
                            "Method-only": method_only, "Init-only": init_only,
                            "McNemar p": f"{exact_mcnemar_pvalue(method_only=method_only, baseline_only=init_only):.6g}",
                        })

    ranks_path = analysis_root / "threshold_ranks.csv"
    with ranks_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Judge", "Target", "Source", "Threshold", "Method", "Rank"))
        writer.writeheader()
        for judge in (primary_judge, SECONDARY_JUDGE):
            for source in sources:
                rates = {
                    threshold: {method: Rate.from_flags(labels[(judge, source, method)][threshold].values()) for method in methods}
                    for threshold in labels[(judge, source, "init")]
                }
                for ranking in threshold_rank_sensitivity(rates):
                    for method, rank in ranking.ranks:
                        writer.writerow({"Judge": judge, "Target": target, "Source": source, "Threshold": ranking.threshold, "Method": method, "Rank": rank})
    payload = {"methods": list(methods), "sources": list(sources), "target_key": target, "samples_per_source": samples_per_source}
    atomic_write_json(analysis_root / "analysis_manifest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--primary-judge", default="octopus_seval_14b")
    args = parser.parse_args()
    print(json.dumps(analyze_selected_matrix(
        parent_root=args.parent_root,
        selection_root=args.selection_root,
        primary_judge=args.primary_judge,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
