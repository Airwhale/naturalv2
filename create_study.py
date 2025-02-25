import logging
import os

import hydra
import yaml
from omegaconf import DictConfig

from naturalv2.evals.clinicaltrials import ClinicalTrial, download_clinical_trials
from naturalv2.evals.experiment import Experiment
from naturalv2.utils import check_trial


logger = logging.getLogger(__name__)


def find_valid_ncts(data_path: str, test: bool = False) -> list[str]:
    stats = {
        "total": 0,
        "randomized": 0,
        "multiple_noncontrol": 0,
        "nonhealthy": 0,
        "binary_endpoint": 0,
    }
    trial_path = os.path.join(data_path, "nct_reports" + ("_test" if test else ""))
    valid_nct_path = os.path.join(trial_path, "valid_binary_nct_ids.txt")

    if not os.path.exists(trial_path):
        download_clinical_trials(trial_path, test)

    if not os.path.exists(valid_nct_path):
        with open(valid_nct_path, "a") as valid_file:
            for filename in os.listdir(trial_path):
                if filename.endswith(".json"):
                    nct_id = filename[:-5]  # ends with .json
                    trial = ClinicalTrial(data_path=trial_path, nct_id=nct_id)
                    trial_stats, check = check_trial(trial)
                    for key, value in trial_stats.items():
                        stats[key] += value
                    if check:
                        valid_file.write(f"{nct_id}\n")
        logger.info("Benchmark Stats: %s", stats)

    with open(valid_nct_path, "r") as valid_file:
        return [line.strip() for line in valid_file.readlines()]


def find_condition_ncts(
    nct_ids: list[str], data_path: str, conditions: list[str], test=False
) -> list[tuple[str, str]]:
    trial_path = os.path.join(data_path, "nct_reports" + ("_test" if test else ""))
    condition_nct_path = os.path.join(
        trial_path, f"valid_binary_{conditions[0]}_nct_ids.txt"
    )
    condition_trials = []
    conditions_set = {cond.replace("_", " ").lower() for cond in conditions}

    for nct_id in nct_ids:
        trial = ClinicalTrial(data_path=trial_path, nct_id=nct_id)
        trial_conditions = {word.lower() for word in trial.conditions + trial.keywords}
        if conditions_set.intersection(trial_conditions):
            result_date = (
                trial.estimated_completion if test else trial.results_first_posted
            )
            condition_trials.append((nct_id, result_date))
            with open(condition_nct_path, "a") as condition_file:
                condition_file.write(f"{nct_id}\n")

    return condition_trials


class Study:
    def __init__(self, retro_trials, test_trials, cfg):
        # order retro_trials by completion date and split into train/val according to train_ratio
        retro_trials.sort(key=lambda x: x[1])
        train_size = int(len(retro_trials) * cfg.train_ratio)
        train_trials, val_trials = retro_trials[:train_size], retro_trials[train_size:]

        self.condition = list(cfg.condition)
        self.train_ratio = cfg.train_ratio
        self.num_train_trials = len(train_trials)
        self.num_val_trials = len(val_trials)
        self.num_test_trials = len(test_trials)

        train_exp = [
            Experiment(nct_id, cfg.eval.data_path, split="train")
            for (nct_id, _) in train_trials
        ]
        self.train_trials = [
            {exp.nct_id: [exp.title, exp.date] + exp.references} for exp in train_exp
        ]
        self.num_train_labels = sum([len(exp.effect_sizes) for exp in train_exp])

        val_exp = [
            Experiment(nct_id, cfg.eval.data_path, split="val")
            for (nct_id, _) in val_trials
        ]
        self.val_trials = [
            {exp.nct_id: [exp.title, exp.date] + exp.references} for exp in val_exp
        ]
        self.num_val_labels = sum([len(exp.effect_sizes) for exp in val_exp])

        test_exp = [
            Experiment(nct_id, cfg.eval.data_path + "_test", split="test")
            for (nct_id, _) in test_trials
        ]
        self.test_trials = [
            {exp.nct_id: [exp.title, exp.date] + exp.references} for exp in test_exp
        ]
        self.num_test_to_predict = sum([len(exp.outcome_treatment) for exp in test_exp])

        print(f"Study created for {self.condition} with:")
        print(f"Train: {len(self.train_trials)} trials, {self.num_train_labels} labels")
        print(f"Val: {len(self.val_trials)} trials, {self.num_val_labels} labels")
        print(
            f"Test: {len(self.test_trials)} trials, {self.num_test_to_predict} to predict"
        )

    def to_yaml(self, filename):
        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # find nct_ids of valid retrospective and test trials
    nct_list = find_valid_ncts(cfg.data_path)
    test_nct_list = find_valid_ncts(cfg.data_path, test=True)
    logger.info(
        "Total valid trials: %s Completed and %s Test",
        len(nct_list),
        len(test_nct_list),
    )

    # find nct_ids of retrospective and test trials related to {condition}
    retro_trials = find_condition_ncts(nct_list, cfg.data_path, cfg.conditions)
    test_trials = find_condition_ncts(
        test_nct_list, cfg.data_path, cfg.conditions, test=True
    )

    study = Study(retro_trials, test_trials, cfg)
    study.to_yaml(os.path.join(cfg.save_path, cfg.condition[0] + "_study.yaml"))

    # TODO: common names + paths to data dumps


if __name__ == "__main__":
    main()
