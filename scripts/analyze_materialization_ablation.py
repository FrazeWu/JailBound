"""Join frozen pair judgments and summarize materialization fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmark.safety_eval.materialization_ablation import (
    BinaryPairJudgment,
    Branch,
    MaterializationPair,
    canonical_pair_key,
    index_pairs,
    write_pair_summaries,
)


def join_judgments(pairs_path: Path, judgments_path: Path, destination: Path) -> tuple[str, ...]:
    pairs = [
        MaterializationPair.model_validate_json(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = index_pairs(pairs)
    judgment_lines = [
        (line_number, line)
        for line_number, line in enumerate(judgments_path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    ]
    if not judgment_lines:
        raise ValueError("paired judgments are empty")
    observed: set[tuple[str, str, Branch, int, str]] = set()
    judge_keys: set[str] = set()
    for line_number, line in judgment_lines:
        try:
            row = json.loads(line)
            source = row["source"]
            sample_id = row["sample_id"]
            branch = Branch(row["branch"])
            checkpoint = row["optimization_checkpoint"]
            judge_key = row["judge_key"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError("source must be non-empty")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError("sample ID must be non-empty")
            if isinstance(checkpoint, bool) or not isinstance(checkpoint, int):
                raise ValueError("optimization checkpoint must be an integer")
            labels = BinaryPairJudgment.model_validate({
                "continuous": row["continuous"],
                "materialized": row["materialized"],
            })
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid paired judgment at line {line_number}") from error
        if not isinstance(judge_key, str) or not judge_key.strip():
            raise ValueError("judge key must be non-empty")

        pair_key = source, sample_id, branch, checkpoint
        record_key = (*pair_key, judge_key)
        if record_key in observed:
            raise ValueError(f"duplicate paired judgment: {record_key}")
        observed.add(record_key)
        pair = indexed.get(pair_key)
        if pair is None:
            raise ValueError(f"judgment references an unknown pair: {pair_key}")
        if pair.status == "failed":
            raise ValueError(f"failed pair cannot receive a judgment: {pair_key}")
        pair.judgments[judge_key] = labels
        judge_keys.add(judge_key)
    complete_keys = {canonical_pair_key(pair) for pair in pairs if pair.status == "complete"}
    for judge_key in judge_keys:
        labeled = {key[:4] for key in observed if key[4] == judge_key}
        if labeled != complete_keys:
            raise ValueError(f"paired judgments are incomplete for {judge_key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(pair.model_dump_json() + "\n" for pair in pairs), encoding="utf-8")
    return tuple(sorted(judge_keys))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    joined = args.output_dir / "materialization_pairs_judged.jsonl"
    judge_keys = join_judgments(args.pairs, args.judgments, joined)
    outputs = write_pair_summaries(joined, args.output_dir, judge_keys=judge_keys)
    print(json.dumps({"judge_keys": list(judge_keys), "outputs": [str(path) for path in outputs]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
