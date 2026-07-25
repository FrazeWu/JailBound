#!/usr/bin/env python3
"""Step 07: Run safety benchmark — baselines, ablations, transferability.

Thin entrypoint over ``scripts/run_dataset_benchmark.py`` so the numbered
pipeline keeps a stable 01→07 interface.

Wires: benchmark.{baseline,ablation,transferability,evaluation}
Output: outputs/results/*.json
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / "run_dataset_benchmark.py"


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"Benchmark driver not found: {TARGET}")
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
