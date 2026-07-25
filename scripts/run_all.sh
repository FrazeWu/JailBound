#!/usr/bin/env bash
# End-to-end pipeline: meta-prompt → optimize → defense fine-tune → benchmark.
set -euo pipefail

python scripts/01_build_meta_prompt.py
bash   scripts/02_train_unalignment.sh
python scripts/03_optimize_attack.py
python scripts/04_build_finetune_data.py
bash   scripts/05_defense_stage1.sh
bash   scripts/06_defense_stage2.sh
python scripts/07_run_benchmark.py
