#!/usr/bin/env bash
# Stage 0 -- fetch run inputs from S3 into SAVE_PATH. No data lives in this repository.
#
# Pulls the study definition and the trial records it references, and lays them out where naturalv2
# expects: studies/, nct_reports/ (completed trials) and nct_reports_test/ (prediction targets).
# The corpus is fetched separately -- it is ~458 MB, see --corpus.
#
#   bash 00_fetch_inputs.sh              # study + trial records (~1 MB)
#   bash 00_fetch_inputs.sh --corpus     # ...and the corpus into PP_CORPUS_DIR
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

S3="${PP_S3_ROOT:-s3://patientpunk/trial_superset}"

command -v aws >/dev/null || { echo "aws CLI not found; see patientpunk/README.md for the layout"; exit 1; }

mkdir -p "$SAVE_PATH/studies" "$SAVE_PATH/nct_reports" "$SAVE_PATH/nct_reports_test"

# The study filename is load-bearing: get_study_filepaths DERIVES it from condition + experiment, so
# a differently-named file is reported as a MISSING study rather than a misnamed one.
aws s3 cp "$S3/core5/long_covid_core5_noparallel_notbinary_apo_study.yaml" \
          "$SAVE_PATH/studies/long_covid_noparallel_notbinary_apo_study.yaml"

# Completed trials (labels) and prospective targets (no labels).
for n in NCT05472090 NCT05047952 NCT05618587 NCT04809974 NCT05874037; do
  aws s3 cp "$S3/m3_labeled/long_covid/nct_reports/$n.json" "$SAVE_PATH/nct_reports/$n.json"
done
aws s3 cp "$S3/relaxed_test/nct_reports_test/NCT06366724.json" "$SAVE_PATH/nct_reports_test/"

if [ "${1:-}" = "--corpus" ]; then
  : "${PP_CORPUS_DIR:?set PP_CORPUS_DIR to where the corpus should live}"
  echo "syncing corpus (~458 MB) -> $PP_CORPUS_DIR"
  aws s3 sync "$S3/natural_corpus_parquet/" "$PP_CORPUS_DIR/"
fi

echo
echo "staged into $SAVE_PATH:"
echo "  studies/           $(ls "$SAVE_PATH/studies" | wc -l) file(s)"
echo "  nct_reports/       $(ls "$SAVE_PATH/nct_reports" | wc -l) trial(s)"
echo "  nct_reports_test/  $(ls "$SAVE_PATH/nct_reports_test" | wc -l) target(s)"
