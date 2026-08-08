#!/usr/bin/env python3
"""Write the reviewer-facing same-state continuous/materialized ASR table."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
import json
from pathlib import Path

from benchmark.safety_eval.materialization_ablation import (
    Branch,
    MaterializationPair,
    canonical_pair_key,
    index_pairs,
)
from benchmark.safety_eval.materialization_judging import (
    PairedJudgmentRecord,
    paired_judgment_key,
)


PRIMARY_JUDGE = "octopus_seval_14b"


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _difference(value: float) -> str:
    return f"{value:+.1f} pp"


def _load_pairs(path: Path) -> tuple[MaterializationPair, ...]:
    rows = tuple(
        MaterializationPair.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    index_pairs(rows)
    return rows


def _load_judgments(path: Path) -> tuple[PairedJudgmentRecord, ...]:
    rows = tuple(
        PairedJudgmentRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    keys = [paired_judgment_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("paired judgments contain duplicate keys")
    return rows


def _cross_branch_projection_collapse(
    pairs: Iterable[MaterializationPair],
) -> dict[str, tuple[int, int]]:
    grouped: dict[tuple[str, str, int], dict[Branch, MaterializationPair]] = defaultdict(dict)
    for pair in pairs:
        grouped[(pair.source, pair.sample_id, pair.optimization_checkpoint)][pair.branch] = pair
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    expected = {Branch.high_value, Branch.safety_sensitivity}
    for (source, _, _), branches in grouped.items():
        if set(branches) != expected:
            continue
        high = branches[Branch.high_value]
        sensitivity = branches[Branch.safety_sensitivity]
        counts[source][1] += 1
        if (
            high.state_sha256 != sensitivity.state_sha256
            and high.projected_token_ids == sensitivity.projected_token_ids
        ):
            counts[source][0] += 1
    return {source: (values[0], values[1]) for source, values in counts.items()}


def build_report(
    pairs: Iterable[MaterializationPair],
    judgments: Iterable[PairedJudgmentRecord],
    *,
    judge_key: str,
) -> str:
    """Validate exact same-state pairs and render the requested five-column table."""
    all_pairs = tuple(pairs)
    indexed_pairs = index_pairs(all_pairs)
    complete_pairs = {
        key: pair for key, pair in indexed_pairs.items() if pair.status == "complete"
    }
    if any(
        pair.materialized_transport != "direct_projected_token_ids"
        for pair in complete_pairs.values()
    ):
        raise ValueError("reviewer table requires direct projected-token transport")

    selected: dict[tuple[str, str, Branch, int], PairedJudgmentRecord] = {}
    provenance: set[tuple[str, str, str, float]] = set()
    for record in judgments:
        if record.judge_key != judge_key:
            continue
        if record.status != "complete":
            raise ValueError("reviewer table requires complete primary judgments")
        key = (
            record.source,
            record.sample_id,
            record.branch,
            record.optimization_checkpoint,
        )
        if key in selected:
            raise ValueError(f"duplicate primary judgment: {key}")
        pair = complete_pairs.get(key)
        if pair is None:
            raise ValueError(f"primary judgment references an unknown complete pair: {key}")
        if record.target_key != pair.model_key or record.target_revision != pair.model_revision:
            raise ValueError(f"primary judgment target differs from pair: {key}")
        selected[key] = record
        provenance.add((record.run_id, record.config_hash, record.judge_revision, record.threshold))
    if set(selected) != set(complete_pairs):
        raise ValueError("primary judgment coverage differs from complete pair coverage")
    if len(provenance) != 1:
        raise ValueError("primary judgments do not share one frozen provenance")

    grouped: dict[tuple[str, Branch | str], list[PairedJudgmentRecord]] = defaultdict(list)
    for key, record in selected.items():
        grouped[(key[0], key[2])].append(record)
    grouped[("Overall", "All")] = list(selected.values())

    lines = [
        "# Same-State Materialization Ablation\n\n",
        "For every pair, Continuous generation uses the saved checkpoint state `e` directly, "
        "whereas Materialized generation uses the exact projected token IDs `Psi(e)`. "
        "The target model, chat scaffold, checkpoint, greedy decoding, and token budget are identical.\n\n",
        f"Primary judge: `{judge_key}`. Difference = Materialized ASR - Continuous ASR, in percentage points.\n\n",
    ]
    collapse = _cross_branch_projection_collapse(complete_pairs.values())
    collapsed = [
        f"{source}: {count}/{matched} matched samples"
        for source, (count, matched) in sorted(collapse.items())
        if count
    ]
    if collapsed:
        lines.extend((
            "**Cross-branch projection collapse:** " + "; ".join(collapsed) + ". ",
            "For these units, distinct saved continuous states map to identical `Psi(e)` token IDs; ",
            "the Materialized response is therefore a shared projected result rather than an independent branch-specific result.\n\n",
        ))
    lines.extend((
        "| Source | Branch | Continuous ASR | Materialized ASR | Difference |\n",
        "|---|---|---:|---:|---:|\n",
    ))
    valid_counts: list[str] = []
    for source, branch in sorted(
        grouped,
        key=lambda value: (value[0] == "Overall", value[0], str(value[1])),
    ):
        rows = grouped[(source, branch)]
        continuous = sum(bool(row.continuous) for row in rows) / len(rows)
        materialized = sum(bool(row.materialized) for row in rows) / len(rows)
        difference = 100.0 * (materialized - continuous)
        branch_text = branch.value if isinstance(branch, Branch) else branch
        lines.append(
            f"| {source} | {branch_text} | {_percent(continuous)} | "
            f"{_percent(materialized)} | {_difference(difference)} |\n"
        )
        if source != "Overall":
            valid_counts.append(f"{source}/{branch_text}: n={len(rows)}")

    failed = len(all_pairs) - len(complete_pairs)
    lines.extend((
        "\n",
        "Valid paired units: " + "; ".join(valid_counts) + ".\n\n",
        f"Excluded before judging: {failed} pair(s) with unavailable or failed checkpoint/generation artifacts.\n",
    ))
    return "".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-key", default=PRIMARY_JUDGE)
    args = parser.parse_args(argv)
    report = build_report(
        _load_pairs(args.pairs),
        _load_judgments(args.judgments),
        judge_key=args.judge_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({"artifact": str(args.output), "judge_key": args.judge_key}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
