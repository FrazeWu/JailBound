#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

while ! tmux has-session -t fol-judge-primary-jailbound 2>/dev/null || ! tmux has-session -t fol-judge-primary-seval 2>/dev/null; do
  sleep 60
done
while tmux has-session -t fol-judge-primary-jailbound 2>/dev/null || tmux has-session -t fol-judge-primary-seval 2>/dev/null; do
  sleep 60
done

tmux new-session -d -s fol-controls-jailbound \
  "cd '$root' && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/run_fol_controls.py --config configs/benchmark/safety_eval_additions.yaml --source jailbound > outputs/fol-controls-jailbound.log 2>&1"
tmux new-session -d -s fol-controls-seval \
  "cd '$root' && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/run_fol_controls.py --config configs/benchmark/safety_eval_additions.yaml --source s_eval > outputs/fol-controls-seval.log 2>&1"
