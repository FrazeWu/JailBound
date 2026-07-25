#!/usr/bin/env bash
# Step 06: Defense stage-2 — auto re-label under-test samples then re-SFT.
#
# Internally: python -m defense.stage2_relabel (relabel) +
# llamafactory-cli train (refit).
set -euo pipefail

CONFIG="${1:-configs/training/defense_stage2.yaml}"
python -m defense.stage2_relabel  # produces relabeled jsonl
ln -sf "$(pwd)/configs/training/llamafactory_dataset_info.json" \
       "$(pwd)/third_party/LLaMA-Factory/data/dataset_info.json"
cd third_party/LLaMA-Factory
exec llamafactory-cli train "../../${CONFIG}"
