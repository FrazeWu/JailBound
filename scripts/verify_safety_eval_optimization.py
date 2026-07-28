"""Fail closed unless every configured main-matrix checkpoint is terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.verification import verify_optimization_matrix


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = verify_optimization_matrix(
        ROOT / config.run.output_root,
        sources=tuple(config.data.sources),
        methods=tuple(config.optimization.methods),
    )
    print(json.dumps({
        "complete": result.complete,
        "expected_records": result.expected_records,
        "observed_records": result.observed_records,
        "errors": list(result.errors),
    }, sort_keys=True))
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
