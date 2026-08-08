"""Join frozen pair judgments and summarize materialization fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmark.safety_eval.materialization_ablation import (
    BinaryPairJudgment,
    MaterializationPair,
    canonical_pair_key,
    index_pairs,
    write_pair_summaries,
)
from benchmark.safety_eval.io import atomic_write_jsonl
from benchmark.safety_eval.materialization_judging import (
    MATCHED_JUDGE_THRESHOLDS,
    PairedJudgmentRecord,
    paired_judgment_key,
)


EXPECTED_JUDGE_THRESHOLDS = MATCHED_JUDGE_THRESHOLDS


def join_judgments(pairs_path: Path, judgments_path: Path, destination: Path) -> tuple[str, ...]:
    pairs = [
        MaterializationPair.model_validate_json(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = index_pairs(pairs)
    if any(pair.judgments for pair in pairs):
        raise ValueError("materialization pairs contain pre-existing judgments")
    judgment_lines = [
        (line_number, line)
        for line_number, line in enumerate(judgments_path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    ]
    if not judgment_lines:
        raise ValueError("paired judgments are empty")
    observed: set[tuple[str, str, object, int, str]] = set()
    judge_keys: set[str] = set()
    config_hashes: set[str] = set()
    run_ids: set[str] = set()
    revisions_by_judge: dict[str, set[str]] = {}
    for line_number, line in judgment_lines:
        try:
            row = json.loads(line)
            if row.get("judge_key") == "":
                raise ValueError("judge key must be non-empty")
            record = PairedJudgmentRecord.model_validate(row)
        except (KeyError, TypeError, ValueError) as error:
            if str(error) == "judge key must be non-empty":
                raise
            raise ValueError(f"invalid paired judgment at line {line_number}: {error}") from error
        if record.status != "complete":
            raise ValueError(f"paired judgment must be terminal complete at line {line_number}")
        expected_threshold = EXPECTED_JUDGE_THRESHOLDS.get(record.judge_key)
        if expected_threshold is None:
            raise ValueError(f"paired judgment does not use the expected judges: {record.judge_key}")
        if record.threshold != expected_threshold:
            raise ValueError(
                f"paired judgment threshold differs for {record.judge_key}: "
                f"{record.threshold} != {expected_threshold}"
            )

        pair_key = record.source, record.sample_id, record.branch, record.optimization_checkpoint
        record_key = paired_judgment_key(record)
        if record_key in observed:
            raise ValueError(f"duplicate paired judgment: {record_key}")
        observed.add(record_key)
        pair = indexed.get(pair_key)
        if pair is None:
            raise ValueError(f"judgment references an unknown pair: {pair_key}")
        if pair.status == "failed":
            raise ValueError(f"failed pair cannot receive a judgment: {pair_key}")
        if record.target_key != pair.model_key or record.target_revision != pair.model_revision:
            raise ValueError(f"judgment target provenance differs from pair: {pair_key}")
        pair.judgments[record.judge_key] = BinaryPairJudgment(
            continuous=record.continuous,
            materialized=record.materialized,
        )
        judge_keys.add(record.judge_key)
        config_hashes.add(record.config_hash)
        run_ids.add(record.run_id)
        revisions_by_judge.setdefault(record.judge_key, set()).add(record.judge_revision)
    if len(config_hashes) != 1:
        raise ValueError("paired judgments contain multiple config_hash values")
    if len(run_ids) != 1:
        raise ValueError("paired judgments contain multiple run_id values")
    for judge_key, revisions in revisions_by_judge.items():
        if len(revisions) != 1:
            raise ValueError(f"paired judgments contain multiple revisions for {judge_key}")
    if judge_keys != set(EXPECTED_JUDGE_THRESHOLDS):
        raise ValueError(
            "paired judgments do not contain the expected judges: "
            f"{sorted(EXPECTED_JUDGE_THRESHOLDS)}"
        )
    complete_keys = {canonical_pair_key(pair) for pair in pairs if pair.status == "complete"}
    for judge_key in EXPECTED_JUDGE_THRESHOLDS:
        labeled = {key[:4] for key in observed if key[4] == judge_key}
        if labeled != complete_keys:
            raise ValueError(f"paired judgments are incomplete for {judge_key}")
    atomic_write_jsonl(destination, (pair.model_dump(mode="json") for pair in pairs))
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
