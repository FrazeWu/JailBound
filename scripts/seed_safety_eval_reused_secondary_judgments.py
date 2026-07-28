"""Seed a replacement gate root with unchanged secondary-judge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.replacement_parent import seed_reused_secondary_judgments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--secondary-judge", default="qwen32_compat")
    args = parser.parse_args()
    output = seed_reused_secondary_judgments(
        original_selection_root=args.original_selection_root,
        output_root=args.output_root,
        target_key=args.target,
        secondary_judge=args.secondary_judge,
    )
    print(json.dumps({"output_root": str(output), "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
