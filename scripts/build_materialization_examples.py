"""Build deterministic full and OpenReview materialization examples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from benchmark.safety_eval.intent_preservation import read_final_labels, read_mapping
from benchmark.safety_eval.materialization_ablation import MaterializationPair
from benchmark.safety_eval.materialization_examples import (
    ExampleSlot,
    build_example_cases,
    render_examples_markdown,
    select_compact_examples,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--final-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append")
    args = parser.parse_args(argv)
    sources = tuple(args.source or ("harmbench", "jailbound", "s_eval"))
    pairs = [
        MaterializationPair.model_validate_json(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mappings = read_mapping(args.mapping)
    labels = read_final_labels(args.final_labels, args.mapping)
    slots = build_example_cases(pairs, mappings, labels, sources=sources)
    compact_cases = select_compact_examples(slots)
    compact_slots = tuple(
        ExampleSlot(case.source, case.branch, case.final_label, case)
        for case in compact_cases
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "materialization_examples_full.md"
    compact_path = args.output_dir / "materialization_examples_openreview.md"
    index_path = args.output_dir / "materialization_example_index.csv"
    full_path.write_text(render_examples_markdown(slots, title="Materialization Examples"), encoding="utf-8")
    compact_path.write_text(render_examples_markdown(compact_slots, title="OpenReview Materialization Examples"), encoding="utf-8")
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("case_id", "source", "sample_id", "branch", "intent_label", "drift_reason", "roundtrip")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for slot in slots:
            case = slot.case
            writer.writerow({
                "case_id": case.case_id if case else "No case available",
                "source": slot.source,
                "sample_id": case.sample_id if case else "",
                "branch": slot.branch.value,
                "intent_label": slot.final_label.value,
                "drift_reason": "|".join(reason.value for reason in case.drift_reasons) if case else "",
                "roundtrip": ("Exact" if case.roundtrip_exact_match else "Changed") if case else "",
            })
    print(json.dumps({
        "compact_case_count": len(compact_cases),
        "compact_markdown": str(compact_path),
        "full_markdown": str(full_path),
        "index_csv": str(index_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
