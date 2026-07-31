"""Validate human intent labels and write count-first IPR artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from benchmark.safety_eval.intent_preservation import (
    DriftReason,
    analyze_intent_labels,
    read_final_labels,
    read_raw_labels,
)
from benchmark.safety_eval.materialization_ablation import MaterializationPair


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument("--final-labels", type=Path, required=True)
    parser.add_argument("--annotator", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    pairs = [
        MaterializationPair.model_validate_json(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = read_raw_labels(args.raw_labels, args.mapping, expected_annotators=tuple(args.annotator))
    final = read_final_labels(args.final_labels, args.mapping)
    analysis = analyze_intent_labels(pairs, args.mapping, raw, final)

    ipr_path = args.output_dir / "intent_preservation.csv"
    _write_csv(
        ipr_path,
        ("Source", "Branch", "Preserved", "Total", "Intent-preservation rate"),
        [{
            "Source": row.source,
            "Branch": row.branch,
            "Preserved": row.preserved,
            "Total": row.total,
            "Intent-preservation rate": row.rate,
        } for row in analysis.ipr],
    )
    reasons_path = args.output_dir / "intent_drift_reasons.csv"
    overall = next(row for row in analysis.ipr if row.source == "Overall")
    not_preserved = overall.total - overall.preserved
    _write_csv(
        reasons_path,
        ("Drift reason", "Count", "Fraction among Not preserved"),
        [{
            "Drift reason": reason.value,
            "Count": analysis.drift_reason_counts.get(reason.value, 0),
            "Fraction among Not preserved": (
                analysis.drift_reason_counts.get(reason.value, 0) / not_preserved if not_preserved else 0.0
            ),
        }
        for reason in DriftReason
        ],
    )
    diagnostics_path = args.output_dir / "intent_annotation_diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps({
        "agreement": asdict(analysis.agreement),
        "duplicate_materialized_prompts": analysis.duplicate_materialized_prompts,
        "judge_cross_tabs": analysis.judge_cross_tabs,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostics": str(diagnostics_path),
        "drift_reasons": str(reasons_path),
        "intent_preservation": str(ipr_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
