#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

while tmux has-session -t fol-generate-jailbound 2>/dev/null || tmux has-session -t fol-generate-seval 2>/dev/null || tmux has-session -t fol-generate-jailbound-0 2>/dev/null || tmux has-session -t fol-generate-jailbound-1 2>/dev/null; do
  sleep 60
done

tmux new-session -d -s fol-judge-primary-jailbound \
  "cd '$root' && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/run_fol_judging.py --config configs/benchmark/safety_eval_additions.yaml --judge primary --source jailbound > outputs/fol-judge-primary-jailbound.log 2>&1"
tmux new-session -d -s fol-judge-primary-seval \
  "cd '$root' && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/run_fol_judging.py --config configs/benchmark/safety_eval_additions.yaml --judge primary --source s_eval > outputs/fol-judge-primary-seval.log 2>&1"
