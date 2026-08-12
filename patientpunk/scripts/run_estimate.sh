#!/usr/bin/env bash
# Stage 3 -- estimate. Generative roles go to an API; probs_model needs a vLLM server exposing
# prompt_logprobs (see ../serving).
#   VLLM_BASE=http://host:port/v1 NCT=NCT05618587 SPLIT=train ./run_estimate.sh
#
# CHEAP is a separate model deliberately: relevance_filter and treatment_outcome_filter score EVERY
# curated report once per outcome, so on a large corpus that coarse pass dominates the bill -- and
# ~98% of what it scores is discarded by the next stage.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GEN=${GEN:-openrouter/google/gemini-2.5-flash}
CHEAP=${CHEAP:-openrouter/google/gemini-2.5-flash-lite}
PROBS_MODEL=${PROBS_MODEL:-Qwen/Qwen2.5-7B-Instruct}
VLLM_BASE=${VLLM_BASE:?set VLLM_BASE to the vLLM endpoint, e.g. http://host:8000/v1}

"$PP_PYTHON" -m naturalv2.cli.estimate_ate \
  --config-name "${CONFIG_NAME:-estimate_mc.yaml}" \
  experiment_name="$EXPERIMENT" \
  "conditions=[\"$CONDITION\"]" \
  nct_id="${NCT:?set NCT, e.g. NCT05618587}" \
  split="${SPLIT:-train}" \
  '~sources.pubmed' \
  'source@sources.reddit=reddit_prebuilt' \
  model@cheap_model=openrouter \
  model@sample_model=openrouter \
  model@imputations_model=openrouter \
  model@probs_model=hosted_vllm \
  cheap_model.model_id="$CHEAP" \
  sample_model.model_id="$GEN" \
  imputations_model.model_id="$GEN" \
  probs_model.model_id="hosted_vllm/$PROBS_MODEL" \
  probs_model.api_base="$VLLM_BASE" \
  probs_model.api_key=EMPTY \
  probs_model.max_parallel_requests="${PROBS_CONCURRENCY:-2}" \
  "$@"
