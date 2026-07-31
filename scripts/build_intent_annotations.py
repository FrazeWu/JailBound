"""Export a deterministic blinded intent-preservation annotation package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmark.safety_eval.intent_preservation import export_blinded_annotations
from benchmark.safety_eval.materialization_ablation import MaterializationPair


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args(argv)
    pairs = [
        MaterializationPair.model_validate_json(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exported = export_blinded_annotations(pairs, seed=args.seed, output_dir=args.output_dir)
    print(json.dumps({
        "annotation_count": len(exported.annotation_ids),
        "blinded_csv": str(exported.blinded_csv),
        "mapping_csv": str(exported.mapping_csv),
        "manifest_json": str(exported.manifest_json),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
