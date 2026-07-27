#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
config="configs/benchmark/reviewer_additions.yaml"
output_root="outputs/results/reviewer_additions_n17_qwen7b_local_qwen32_compat_eager_randomfix"

cd "$root"
fol_ready=0
fol_interpolation_ready=0
while tmux has-session -t reviewer-n17-gpu0 2>/dev/null || tmux has-session -t reviewer-n17-gpu1 2>/dev/null; do
  sleep 60
done

uv run python scripts/verify_reviewer_optimization.py --config "$config" \
  > outputs/reviewer-n17-optimization-verification.log 2>&1

CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_fol_manifests.py --config "$config" \
  > outputs/reviewer-n17-fol-manifests.log 2>&1

tmux new-session -d -s reviewer-n17-fol-gpu0 -c "$root" \
  "CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_fol_candidate_optimization.py --config $config --source jailbound > outputs/reviewer-n17-fol-gpu0.log 2>&1"
tmux new-session -d -s reviewer-n17-fol-gpu1 -c "$root" \
  "CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_fol_candidate_optimization.py --config $config --source s_eval > outputs/reviewer-n17-fol-gpu1.log 2>&1"
while tmux has-session -t reviewer-n17-fol-gpu0 2>/dev/null || tmux has-session -t reviewer-n17-fol-gpu1 2>/dev/null; do
  sleep 60
done

if ! CUDA_VISIBLE_DEVICES=0 uv run python scripts/select_fol_validation.py --config "$config" \
  > outputs/reviewer-n17-fol-selection.log 2>&1; then
  printf '%s\n' 'inconclusive' > "$output_root/fol_boundary/FOL_SELECTION_STATUS"
else
  uv run python scripts/build_fol_perturbation_schedule.py --config "$config" \
    > outputs/reviewer-n17-fol-schedule.log 2>&1
fi

CUDA_VISIBLE_DEVICES=0 uv run python scripts/freeze_semantic_calibration.py --config "$config" --output-root "$output_root" \
  > outputs/reviewer-n17-semantic-calibration.log 2>&1
if CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_fol_radius_calibration.py --config "$config" \
  > outputs/reviewer-n17-fol-radius-calibration.log 2>&1; then
  uv run python scripts/build_fol_validation_schedule.py --config "$config" \
    > outputs/reviewer-n17-fol-validation-schedule.log 2>&1
  tmux new-session -d -s reviewer-n17-fol-semantic-gpu0 -c "$root" \
    "CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_fol_radius_calibration.py --config $config --schedule-name perturbation_schedule.jsonl --outcomes-name perturbation_semantic_outcomes.jsonl --source jailbound --skip-base-radius > outputs/reviewer-n17-fol-semantic-gpu0.log 2>&1"
  tmux new-session -d -s reviewer-n17-fol-semantic-gpu1 -c "$root" \
    "CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_fol_radius_calibration.py --config $config --schedule-name perturbation_schedule.jsonl --outcomes-name perturbation_semantic_outcomes.jsonl --source s_eval --skip-base-radius > outputs/reviewer-n17-fol-semantic-gpu1.log 2>&1"
  while tmux has-session -t reviewer-n17-fol-semantic-gpu0 2>/dev/null || tmux has-session -t reviewer-n17-fol-semantic-gpu1 2>/dev/null; do
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_fol_generation.py --config "$config" \
    > outputs/reviewer-n17-fol-generation.log 2>&1
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_fol_judging.py --config "$config" --judge primary \
    > outputs/reviewer-n17-fol-primary-judge.log 2>&1
  fol_ready=1
  if ! CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_fol_controls.py --config "$config" \
    > outputs/reviewer-n17-fol-controls.log 2>&1; then
    printf '%s\n' 'inconclusive' > "$output_root/fol_boundary/FOL_CONTROLS_STATUS"
  fi
  if CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_fol_interpolation.py --config "$config" \
    > outputs/reviewer-n17-fol-interpolation.log 2>&1; then
    if CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_fol_judging.py --config "$config" --judge primary --method fol_interpolation \
      > outputs/reviewer-n17-fol-interpolation-primary-judge.log 2>&1; then
      fol_interpolation_ready=1
    else
      printf '%s\n' 'inconclusive' > "$output_root/fol_boundary/FOL_INTERPOLATION_PRIMARY_JUDGE_STATUS"
    fi
  else
    printf '%s\n' 'inconclusive' > "$output_root/fol_boundary/FOL_INTERPOLATION_STATUS"
  fi
else
  printf '%s\n' 'inconclusive' > "$output_root/fol_boundary/FOL_RADIUS_STATUS"
fi
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_reviewer_experiments.py materialize --config "$config" --output-root "$output_root" --final-only \
  > outputs/reviewer-n17-materialize.log 2>&1

HF_MODULES_CACHE="$root/outputs/hf_modules_cache" CUDA_VISIBLE_DEVICES=1 \
  uv run -m vllm.entrypoints.openai.api_server \
  --model /home/data0/llm_datasets/qwen/qwen3-32b-awq \
  --served-model-name qwen3-32b-awq \
  --host 0.0.0.0 --port 8001 --trust-remote-code --max-model-len 8192 \
  --enforce-eager --gpu-memory-utilization 0.85 --disable-log-requests \
  > outputs/qwen3-32b-awq-gpu1.log 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > outputs/qwen3-32b-awq-gpu1.pid

for _ in {1..120}; do
  if curl -fsS http://127.0.0.1:8001/v1/models >/dev/null; then
    break
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8001/v1/models >/dev/null

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_reviewer_experiments.py run-target --config "$config" --output-root "$output_root" --target qwen2_5_7b \
  > outputs/reviewer-n17-target-qwen7b.log 2>&1
if [ "$fol_ready" -eq 1 ]; then
  uv run python scripts/run_fol_judging.py --config "$config" --judge secondary \
    > outputs/reviewer-n17-fol-secondary-judge.log 2>&1
  uv run python scripts/analyze_fol_boundary.py --config "$config" \
    > outputs/reviewer-n17-fol-analysis.log 2>&1
fi
uv run python scripts/run_reviewer_experiments.py analyze --config "$config" --output-root "$output_root" \
  > outputs/reviewer-n17-analysis.log 2>&1
