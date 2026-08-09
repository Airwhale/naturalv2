"""Build per-trial experiment YAML files for an existing (hand-curated) study.yaml.

Use when a study.yaml already lists which NCT IDs belong to train/val/test but
the corresponding experiments/{experiment_name}/{nct_id}.yaml files were never
generated (e.g. the study.yaml was hand-edited rather than produced by
create_study). Requires the trial JSON for each NCT ID to already exist under
{save_path}/nct_reports (train/val) or {save_path}/nct_reports_test (test).

Usage: python -m scripts.build_experiments_from_study \
    --study_yaml outputs/studies/long_covid_core5_noparallel_notbinary_apo_study.yaml \
    --save_path outputs
"""

import argparse
import logging

from naturalv2.experiment import Experiment
from naturalv2.study import Study
from naturalv2.utils import get_experiment_filepath


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_experiments(study: Study, save_path: str, require_binary_endpoint: bool) -> None:
    for trials, status in [
        (study.train_trials, "completed"),
        (study.val_trials, "completed"),
        (study.test_trials, "active"),
    ]:
        for trial in trials:
            (nct_id,) = trial.keys()
            logger.info("Building experiment for %s (status=%s)", nct_id, status)
            exp = Experiment(
                save_path,
                nct_id,
                study.experiment_name,
                status=status,
                require_binary_endpoint=require_binary_endpoint,
            )
            exp.to_yaml(get_experiment_filepath(save_path, nct_id, study.experiment_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build experiment YAML files for a hand-curated study.yaml"
    )
    parser.add_argument("--study_yaml", required=True, help="Path to the study.yaml")
    parser.add_argument(
        "--save_path",
        required=True,
        help="Root dir containing nct_reports[_test] and where experiments/ will be written",
    )
    parser.add_argument(
        "--require_binary_endpoint",
        action="store_true",
        help="Only include primary outcomes with a binary endpoint "
        "(should match the trial_filters.binary_endpoint used when curating the study)",
    )
    args = parser.parse_args()

    study = Study.from_yaml(args.study_yaml)
    build_experiments(study, args.save_path, args.require_binary_endpoint)
