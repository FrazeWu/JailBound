#!/usr/bin/env bash
# Step 05: Defense stage-1 — initial safety alignment SFT on (clean + malicious).
set -euo pipefail

CONFIG="${1:-configs/training/defense_stage1.yaml}"
ln -sf "$(pwd)/configs/training/llamafactory_dataset_info.json" \
       "$(pwd)/third_party/LLaMA-Factory/data/dataset_info.json"
cd third_party/LLaMA-Factory
exec llamafactory-cli train "../../${CONFIG}"
