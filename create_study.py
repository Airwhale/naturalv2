import os
import yaml
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np

from naturalv2.utils import check_trial
from naturalv2.evals.experiment import Experiment

def find_valid_ncts(cfg, test=False):
    trial_path = cfg.data_path + "/nct_reports"
    if test:
        trial_path += "_test"
    valid_nct_path = trial_path + '/valid_binary_nct_ids.txt'
    if not os.path.exists(valid_nct_path):
        for filename in os.listdir(trial_path):
            if filename.endswith('.json'):
                nct_id = filename[:-5] # ends with .json
                trial = instantiate(cfg.eval, data_path=trial_path, nct_id=nct_id)
                if check_trial(trial):
                    with open(valid_nct_path, 'a') as f:
                        f.write(f"{nct_id}\n")          
    with open(valid_nct_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def find_indication_ncts(nct_list, cfg, test=False):
    trial_path = cfg.data_path + "/nct_reports"
    if test:
        trial_path += "_test"
    indication_nct_path = trial_path + '/valid_binary_{cfg.indication}_nct_ids.txt'
    indication_trials = []
    for nct_id in nct_list:
        trial = instantiate(cfg.eval, data_path=trial_path, nct_id=nct_id)
        # check if {indication} is mentioned in the trial's list of conditions or keywords
        indication = cfg.indication.replace("_", " ")
        if any(indication.lower() in word.lower() for word in trial.conditions + trial.keywords):
            result_date = trial.estimated_completion if test else trial.results_first_posted
            indication_trials.append((nct_id, result_date))
            with open(indication_nct_path, 'a') as f:
                f.write(f"{nct_id}\n")
    return indication_trials

class Study:
    def __init__(self, retro_trials, test_trials, cfg):
        # order retro_trials by completion date and split into train/val according to train_ratio
        retro_trials.sort(key=lambda x: x[1])
        train_size = int(len(retro_trials) * cfg.train_ratio) 
        self.train_trials, self.val_trials = retro_trials[:train_size], retro_trials[train_size:]
        self.test_trials = test_trials

        self.indication = cfg.indication
        self.train_ratio = cfg.train_ratio
        
        train_exp = [Experiment(nct_id, cfg.eval.data_path, train=True) for (nct_id, _) in self.train_trials]
        self.num_train_labels = sum([len(exp.effect_sizes) for exp in train_exp])
        val_exp = [Experiment(nct_id, cfg.eval.data_path, train=False) for (nct_id, _) in self.val_trials]
        self.num_val_labels = sum([len(exp.effect_sizes) for exp in val_exp])

        print(f"Study created for {self.indication} with:") 
        print(f"Train: {len(self.train_trials)} trials, {self.num_train_labels} labels")
        print(f"Val: {len(self.val_trials)} trials, {self.num_val_labels} labels")
        print(f"Test: {len(self.test_trials)} trials")

    def to_yaml(self, filename):
        with open(filename, "w") as file:
            yaml.dump(self.__dict__, file)


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    # find nct_ids of valid retrospective and test trials
    nct_list = find_valid_ncts(cfg)
    test_nct_list = find_valid_ncts(cfg, test=True)

    # find nct_ids of retrospective and test trials related to indication
    retro_trials = find_indication_ncts(nct_list, cfg)
    test_trials = find_indication_ncts(test_nct_list, cfg, test=True)
    
    study = Study(retro_trials, test_trials, cfg)
    study.to_yaml(os.path.join(cfg.save_path, cfg.indication + "_study.yaml"))

    # TODO: paths to data dumps

if __name__ == "__main__":
    main()