#!/usr/bin/env python3
"""Export dataset-grouped intent-preservation annotation tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FIELDNAMES = (
    "sample_id",
    "branch",
    "reference_intent",
    "materialized_prompt",
    "human_annotation",
)
SOURCE_FILENAMES = {
    "harmbench": "harmbench_annotation.csv",
    "jailbound": "jailbound_annotation.csv",
    "s_eval": "s_eval_annotation.csv",
}
BRANCH_ORDER = {"High-Value": 0, "Safety-Sensitivity": 1}


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    failures: list[dict[str, object]] = []

    for pair in pairs:
        if pair["status"] != "complete":
            failures.append(
                {
                    "source": pair["source"],
                    "sample_id": pair["sample_id"],
                    "branch": pair["branch"],
                    "error": pair.get("error") or "unknown error",
                }
            )
            continue
        grouped[pair["source"]].append(
            {
                "sample_id": pair["sample_id"],
                "branch": pair["branch"],
                "reference_intent": pair["reference_intent"],
                "materialized_prompt": pair["materialized_text"],
                "human_annotation": "",
            }
        )

    unknown_sources = set(grouped) - set(SOURCE_FILENAMES)
    if unknown_sources:
        raise ValueError(f"unsupported sources: {sorted(unknown_sources)}")
    for source, filename in SOURCE_FILENAMES.items():
        rows = sorted(
            grouped[source],
            key=lambda row: (str(row["sample_id"]), BRANCH_ORDER[str(row["branch"])]),
        )
        _write_csv(args.output_dir / filename, FIELDNAMES, rows)

    failures.sort(
        key=lambda row: (
            str(row["source"]),
            str(row["sample_id"]),
            BRANCH_ORDER[str(row["branch"])],
        )
    )
    _write_csv(
        args.output_dir / "failed_units.csv",
        ("source", "sample_id", "branch", "error"),
        failures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
