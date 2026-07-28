#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU0_QUEUE_PID GPU1_QUEUE_PID" >&2
  exit 2
fi

gpu0_queue_pid="$1"
gpu1_queue_pid="$2"
root="$(cd "$(dirname "$0")/.." && pwd)"
output_root="outputs/results/reviewer_additions_n17_qwen7b_local_qwen32_compat_eager_randomfix"

cd "$root"
while kill -0 "$gpu0_queue_pid" 2>/dev/null || kill -0 "$gpu1_queue_pid" 2>/dev/null; do
  sleep 60
done

CUDA_VISIBLE_DEVICES=0 uv run python scripts/freeze_semantic_calibration.py \
  --config configs/benchmark/safety_eval_additions.yaml \
  --output-root "$output_root" \
  > outputs/safety-eval-semantic-calibration.log 2>&1

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_safety_eval.py materialize \
  --config configs/benchmark/safety_eval_additions.yaml \
  --output-root "$output_root" \
  --final-only \
  > outputs/safety-eval-materialize.log 2>&1

HF_MODULES_CACHE="$root/outputs/hf_modules_cache" CUDA_VISIBLE_DEVICES=1 \
  uv run -m vllm.entrypoints.openai.api_server \
    --model /home/data0/llm_datasets/qwen/qwen3-32b-awq \
    --served-model-name qwen3-32b-awq \
    --host 0.0.0.0 \
    --port 8001 \
    --trust-remote-code \
    --max-model-len 8192 \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --disable-log-requests \
    > outputs/qwen3-32b-awq-gpu1.log 2>&1 &
server_pid=$!

ready=false
for _ in {1..120}; do
  if curl -fsS http://127.0.0.1:8001/v1/models >/dev/null; then
    ready=true
    break
  fi
  sleep 5
done
if [[ "$ready" != true ]]; then
  kill "$server_pid" 2>/dev/null || true
  exit 1
fi
printf '%s\n' "$server_pid" > outputs/qwen3-32b-awq-gpu1.pid

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_safety_eval.py run-target \
  --config configs/benchmark/safety_eval_additions.yaml \
  --output-root "$output_root" \
  --target qwen2_5_7b \
  > outputs/safety-eval-target-qwen7b.log 2>&1
