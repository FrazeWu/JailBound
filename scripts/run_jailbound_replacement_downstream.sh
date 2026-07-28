#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

config="configs/benchmark/safety_eval_jailbound_replacement.yaml"
replacement_root="outputs/results/reviewer_additions_n17_jailbound_replacement"
original_root="outputs/results/reviewer_additions_n17_qwen7b_local_qwen32_compat_eager_randomfix"
original_selection="$original_root/selected_matrix_3src_7methods"
combined_root="outputs/results/reviewer_additions_n17_selected_matrix_replacement_parent"
selection_root="$replacement_root/selected_matrix_3src_7methods_replacement"
target="qwen2_5_7b"
methods=(init gcg pez gbda zol jailbound_o_minus jailbound_o_plus)

while true; do
  complete=1
  for method in "${methods[@]}"; do
    expected=68
    if [[ "$method" == "init" ]]; then
      expected=17
    fi
    path="$replacement_root/optimization/jailbound/$method/records.jsonl"
    actual=0
    if [[ -f "$path" ]]; then
      actual="$(wc -l < "$path")"
    fi
    if [[ "$actual" -ne "$expected" ]]; then
      complete=0
    fi
  done
  if [[ "$complete" -eq 1 ]]; then
    break
  fi
  if ! tmux has-session -t jailbound-replacement-7method 2>/dev/null; then
    printf '%s\n' 'replacement optimization ended before all required terminal records were written' >&2
    exit 1
  fi
  sleep 60
done

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/run_safety_eval.py materialize \
  --config "$config" --final-only

target_args=(run-target --config "$config" --target "$target" --source jailbound)
for method in "${methods[@]}"; do
  target_args+=(--method "$method")
done
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/run_safety_eval.py "${target_args[@]}"

PYTHONPATH=src .venv/bin/python scripts/build_safety_eval_replacement_parent.py \
  --original-root "$original_root" \
  --replacement-root "$replacement_root" \
  --output-root "$combined_root" \
  --target "$target"

PYTHONPATH=src .venv/bin/python scripts/seed_safety_eval_reused_secondary_judgments.py \
  --original-selection-root "$original_selection" \
  --output-root "$selection_root" \
  --target "$target"

PYTHONPATH=src .venv/bin/python scripts/run_safety_eval_method_gated.py \
  --config configs/benchmark/safety_eval_additions.yaml \
  --parent-root "$combined_root" \
  --selection-root "$selection_root" \
  --target "$target"

PYTHONPATH=src .venv/bin/python scripts/analyze_safety_eval_selected_matrix.py \
  --parent-root "$combined_root" \
  --selection-root "$selection_root"
