import os
import yaml
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np

from naturalv2.utils import check_trial
from naturalv2.evals.experiment import Experiment

def find_valid_ncts(cfg, test=False):
    stats = {
        'total': 0,
        'randomized': 0,
        'multiple_noncontrol': 0,
        'nonhealthy': 0,
        'binary_endpoint': 0
    }
    trial_path = cfg.data_path + "/nct_reports"
    if test:
        trial_path += "_test"
    valid_nct_path = trial_path + '/valid_binary_nct_ids.txt'
    if not os.path.exists(valid_nct_path):
        for filename in os.listdir(trial_path):
            if filename.endswith('.json'):
                nct_id = filename[:-5] # ends with .json
                trial = instantiate(cfg.eval, data_path=trial_path, nct_id=nct_id)
                trial_stats, check = check_trial(trial)
                for key, value in trial_stats.items():
                    stats[key] += value
                if check:
                    with open(valid_nct_path, 'a') as f:
                        f.write(f"{nct_id}\n") 
        print("Benchmark Stats", stats)
    with open(valid_nct_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def find_condition_ncts(nct_list, cfg, test=False):
    trial_path = cfg.data_path + "/nct_reports"
    if test:
        trial_path += "_test"
    condition_nct_path = trial_path + f'/valid_binary_{cfg.condition[0]}_nct_ids.txt'
    condition_trials = []
    for nct_id in nct_list:
        trial = instantiate(cfg.eval, data_path=trial_path, nct_id=nct_id)
        # check if any of {conditions} is mentioned in the trial's list of conditions or keywords
        conditions = [cond.replace("_", " ") for cond in cfg.condition]
        if any(
            any(
                cond.lower() in word.lower() 
                for word in trial.conditions + trial.keywords
            ) 
            for cond in conditions
        ):
            result_date = trial.estimated_completion if test else trial.results_first_posted
            condition_trials.append((nct_id, result_date))
            with open(condition_nct_path, 'w') as f:
                f.write(f"{nct_id}\n")
    return condition_trials

class Study:
    def __init__(self, retro_trials, test_trials, cfg):
        # order retro_trials by completion date and split into train/val according to train_ratio
        retro_trials.sort(key=lambda x: x[1])
        train_size = int(len(retro_trials) * cfg.train_ratio) 
        train_trials, val_trials = retro_trials[:train_size], retro_trials[train_size:]
        test_trials = test_trials

        self.condition = list(cfg.condition)
        self.train_ratio = cfg.train_ratio
        self.num_train_trials = len(train_trials)
        self.num_val_trials = len(val_trials)
        self.num_test_trials = len(test_trials)
        
        train_exp = [Experiment(nct_id, cfg.eval.data_path, split='train') for (nct_id, _) in train_trials]
        self.train_trials = [{exp.nct_id: [exp.title, exp.date] + exp.references} for exp in train_exp]
        self.num_train_labels = sum([len(exp.effect_sizes) for exp in train_exp])
        
        val_exp = [Experiment(nct_id, cfg.eval.data_path, split='val') for (nct_id, _) in val_trials]
        self.val_trials = [{exp.nct_id: [exp.title, exp.date] + exp.references} for exp in val_exp]
        self.num_val_labels = sum([len(exp.effect_sizes) for exp in val_exp])

        test_exp = [Experiment(nct_id, cfg.eval.data_path + '_test', split='test') for (nct_id, _) in test_trials]
        self.test_trials = [{exp.nct_id: [exp.title, exp.date] + exp.references} for exp in test_exp]
        self.num_test_to_predict = sum([len(exp.outcome_treatment) for exp in test_exp])

        print(f"Study created for {self.condition} with:") 
        print(f"Train: {len(self.train_trials)} trials, {self.num_train_labels} labels")
        print(f"Val: {len(self.val_trials)} trials, {self.num_val_labels} labels")
        print(f"Test: {len(self.test_trials)} trials, {self.num_test_to_predict} to predict")

    def to_yaml(self, filename):
        with open(filename, "w") as file:
            yaml.safe_dump(self.__dict__, file)


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    # find nct_ids of valid retrospective and test trials
    nct_list = find_valid_ncts(cfg)
    test_nct_list = find_valid_ncts(cfg, test=True)
    print(f"Total valid trials: {len(nct_list)} Completed and {len(test_nct_list)} Test")

    # find nct_ids of retrospective and test trials related to {condition}
    retro_trials = find_condition_ncts(nct_list, cfg)
    test_trials = find_condition_ncts(test_nct_list, cfg, test=True)
    
    study = Study(retro_trials, test_trials, cfg)
    study.to_yaml(os.path.join(cfg.save_path, cfg.condition[0] + "_study.yaml"))

    # TODO: common names + paths to data dumps 

if __name__ == "__main__":
    main()