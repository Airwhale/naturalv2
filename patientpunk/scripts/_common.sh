#!/usr/bin/env bash
# Shared setup for the patientpunk run scripts. Sourced, not executed.
#
# Everything resolves from the repo, so the scripts work from any checkout with no edits:
#   PP_PYTHON     interpreter to use            (default: python)
#   PP_ENV_FILE   file of API keys to source    (default: <repo>/.env, skipped if absent)
#   PP_CORPUS_DIR the pre-built parquet corpus  (required by the reddit_prebuilt source)
#   SAVE_PATH     naturalv2 output root         (default: <repo>/outputs)
#   EXPERIMENT    experiment/preset name        (default: noparallel_notbinary)
#   CONDITION     condition string              (default: Long Covid)

PP_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PP_SCRIPTS/../.." && pwd)"
cd "$REPO"

PP_ENV_FILE="${PP_ENV_FILE:-$REPO/.env}"
if [ -f "$PP_ENV_FILE" ]; then
  set -a; . "$PP_ENV_FILE"; set +a
fi

export SAVE_PATH="${SAVE_PATH:-$REPO/outputs}"
PP_PYTHON="${PP_PYTHON:-python}"
EXPERIMENT="${EXPERIMENT:-noparallel_notbinary}"
CONDITION="${CONDITION:-Long Covid}"
