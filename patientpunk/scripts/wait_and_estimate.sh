#!/usr/bin/env bash
# Wait for a vLLM endpoint to serve, probe it with a realistic-length prompt_logprobs request, then
# estimate. The probe fails in seconds if the serving config cannot handle real prompts, instead of
# burning a whole pipeline pass to find out -- a tiny prompt always passes and proves nothing.
#   ./wait_and_estimate.sh http://host:port/v1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_common.sh"

BASE=${1:?usage: wait_and_estimate.sh http://host:port/v1}

echo "waiting for vLLM at $BASE ..."
for _ in $(seq 1 100); do
  if curl -sf --max-time 10 "$BASE/models" | grep -q '"id"'; then echo "server READY"; break; fi
  sleep 15
done
curl -sf --max-time 10 "$BASE/models" | grep -q '"id"' || { echo "NEVER READY"; exit 1; }

echo "--- probe: realistic-length prompt_logprobs ---"
if ! "$PP_PYTHON" "$HERE/../serving/probe_long_logprobs.py" "$BASE"; then
  echo "ABORTING before the estimate: this server cannot do prompt_logprobs on real prompts."
  exit 1
fi

# A crashed stage leaves a 0-byte CSV that the resume logic then reads -> EmptyDataError.
find "$SAVE_PATH/results" -size 0 -name "*.csv" -delete 2>/dev/null || true

echo "--- running estimate ---"
VLLM_BASE="$BASE" bash "$HERE/run_estimate.sh" 2>&1 | grep -viE "^[[:space:]]*$|it/s\]|s/it\]"
