#!/usr/bin/env bash
# Stage 2 -- build the per-trial evidence pool from the pre-built corpus. No LLM, no cost.
#   NCT=NCT05618587 SPLIT=train ./run_filter_curate.sh
# Pass --cfg job to print the resolved config without running.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

: "${PP_CORPUS_DIR:?set PP_CORPUS_DIR to the pre-built parquet corpus}"

"$PP_PYTHON" -m naturalv2.cli.filter_curate \
  experiment_name="$EXPERIMENT" \
  "conditions=[\"$CONDITION\"]" \
  nct_id="${NCT:?set NCT, e.g. NCT05618587}" \
  split="${SPLIT:-train}" \
  'source@sources.reddit=reddit_prebuilt' \
  '~sources.pubmed' \
  "$@"
