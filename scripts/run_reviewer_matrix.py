"""Run the locked reviewer optimization matrix serially and resumably."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from benchmark.reviewer_eval.config import load_config
from benchmark.reviewer_eval.io import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def _records_complete(root: Path, source: str, method: str, samples: int) -> bool:
    rows = read_jsonl(root / "optimization" / source / method / "records.jsonl")
    required = {0} if method == "init" else {0, 25, 50, 100}
    checkpoints: dict[str, set[int]] = {}
    for row in rows:
        sample_id, checkpoint = row.get("sample_id"), row.get("checkpoint")
        if isinstance(sample_id, str) and isinstance(checkpoint, int):
            checkpoints.setdefault(sample_id, set()).add(checkpoint)
    return len(checkpoints) == samples and all(values == required for values in checkpoints.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--source", action="append")
    parser.add_argument("--method", action="append")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    sources = _selected_values(args.source, config.data.sources, field="source")
    methods = _selected_values(args.method, config.optimization.methods, field="method")
    for source in sources:
        for method in methods:
            if _records_complete(root, source, method, config.data.samples_per_source):
                print(f"skip complete {source} {method}", flush=True)
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_reviewer_experiments.py"),
                "optimize",
                "--config",
                str(args.config),
                "--output-root",
                str(args.output_root),
                "--source",
                source,
                "--method",
                method,
            ]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
            result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if result.returncode:
                return result.returncode
    return 0


def _selected_values(
    requested: Sequence[str] | None,
    configured: Sequence[str],
    *,
    field: str,
) -> tuple[str, ...]:
    if not requested:
        return tuple(configured)
    unknown = sorted(set(requested) - set(configured))
    if unknown:
        raise ValueError(f"unknown {field}: {', '.join(unknown)}")
    selected = set(requested)
    return tuple(value for value in configured if value in selected)


if __name__ == "__main__":
    raise SystemExit(main())
