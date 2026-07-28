#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

while tmux has-session -t fol-generate-seval 2>/dev/null; do
  sleep 60
done

tmux kill-session -t fol-generate-jailbound 2>/dev/null || true
sleep 5

tmux new-session -d -s fol-generate-jailbound-0 \
  "cd '$root' && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/run_fol_generation.py --config configs/benchmark/safety_eval_additions.yaml --source jailbound --shard-index 0 --shard-count 2 > outputs/fol-generate-jailbound-0.log 2>&1"
tmux new-session -d -s fol-generate-jailbound-1 \
  "cd '$root' && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/run_fol_generation.py --config configs/benchmark/safety_eval_additions.yaml --source jailbound --shard-index 1 --shard-count 2 > outputs/fol-generate-jailbound-1.log 2>&1"
