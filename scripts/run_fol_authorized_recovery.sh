#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

config="configs/benchmark/safety_eval_additions.yaml"
result_root="outputs/results/reviewer_additions_n17_qwen7b_local_qwen32_compat_eager_randomfix"
selection_root="outputs/results/reviewer_additions_n17_jailbound_replacement/selected_matrix_3src_7methods_replacement"

while tmux has-session -t jailbound-replacement-downstream 2>/dev/null; do
  sleep 60
done

test -f "$selection_root/analysis/analysis_manifest.json"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  .venv/bin/python scripts/run_fol_candidate_optimization.py \
  --config "$config" \
  --source jailbound \
  --method jailbound_o_plus_recovery_fd_sdpa \
  --sample-id-file "$result_root/fol_boundary/manifests/jailbound_oom_recovery_ids.json" \
  --shard-index 1 \
  --shard-count 2 \
  --attention-backend sdpa

for shard_index in 0 1 2; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
    .venv/bin/python scripts/run_fol_candidate_optimization.py \
    --config "$config" \
    --source s_eval \
    --method jailbound_o_plus_recovery_fd_sdpa \
    --sample-id-file "$result_root/fol_boundary/manifests/s_eval_oom_recovery_ids.json" \
    --shard-index "$shard_index" \
    --shard-count 3 \
    --attention-backend sdpa
done

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/select_fol_validation.py --config "$config"
