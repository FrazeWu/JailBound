#!/usr/bin/env bash
# Step 02: Train unalignment model M_a via LLaMA-Factory (third_party submodule).
#
# Prereq: `git submodule update --init third_party/LLaMA-Factory`
#         `pip install -e "third_party/LLaMA-Factory[torch,metrics]"`
#
# Usage: bash scripts/02_train_unalignment.sh [config_yaml]
set -euo pipefail

CONFIG="${1:-configs/training/qwen3_4BI_lora_sft.yaml}"

# Symlink LLaMA-Factory's dataset_info.json so the trainer sees our custom entries.
ln -sf "$(pwd)/configs/training/llamafactory_dataset_info.json" \
       "$(pwd)/third_party/LLaMA-Factory/data/dataset_info.json"

cd third_party/LLaMA-Factory
exec llamafactory-cli train "../../${CONFIG}"
