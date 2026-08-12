#!/usr/bin/env bash
# Generic vLLM OpenAI-server launcher for Dispersed.
#
# Dispersed can't override the container command, but it CAN pass env vars — so the served model
# and options are selected entirely via env at job-launch time. One image serves ANY model (7B
# smoke, 72B real, quantized, ...) with no rebuild: set MODEL (and optionally QUANTIZATION,
# MAX_MODEL_LEN, ...) in the job's `env` field.
set -euo pipefail

: "${MODEL:?MODEL env is required, e.g. Qwen/Qwen2.5-7B-Instruct}"

args=(
  --model "$MODEL"
  --served-model-name "$MODEL"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}"
  --host 0.0.0.0
  --port "${PORT:-8000}"
)
[ -n "${QUANTIZATION:-}" ] && args+=(--quantization "$QUANTIZATION")
if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra=(${EXTRA_ARGS})
  args+=("${extra[@]}")
fi

echo "vLLM launch: python3 -m vllm.entrypoints.openai.api_server ${args[*]}"
exec python3 -m vllm.entrypoints.openai.api_server "${args[@]}"
