#!/usr/bin/env python3
"""Step 04: Build defense-side fine-tuning dataset.

Constructs three-class samples (clean / under-test / malicious) for the two-stage
iterative defense (see paper §4).

Wires: defense.data_builder
Output: data/sft/defense_stage{1,2}_*.jsonl

Defense data construction is intentionally not implemented yet. This script
provides the stable CLI surface for the future implementation and fails with a
clear error instead of silently producing incomplete data.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build defense fine-tuning data (stub).")
    p.add_argument("--clean-data", type=Path, required=True, help="Clean sample JSONL input.")
    p.add_argument("--attack-data", type=Path, required=True, help="Optimized attack JSONL input.")
    p.add_argument("--output", type=Path, required=True, help="Output defense SFT JSONL path.")
    p.add_argument(
        "--under-test-ratio",
        type=float,
        default=0.2,
        help="Fraction reserved as under-test samples for future relabeling (stub only).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    raise NotImplementedError(
        "Defense data construction is not implemented yet. "
        "Expected future behavior: read clean samples from "
        f"{args.clean_data}, read optimized attacks from {args.attack_data}, "
        "build clean/under-test/malicious SFT records, and write "
        f"{args.output}."
    )


if __name__ == "__main__":
    main()
