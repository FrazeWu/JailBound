"""Build a fail-closed parent view containing the replacement JailBound run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.safety_eval.replacement_parent import assemble_replacement_parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--primary-judge", default="octopus_seval_14b")
    args = parser.parse_args()
    output = assemble_replacement_parent(
        original_root=args.original_root,
        replacement_root=args.replacement_root,
        output_root=args.output_root,
        target_key=args.target,
        primary_judge=args.primary_judge,
    )
    print(json.dumps({"output_root": str(output), "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
