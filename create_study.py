import os
import yaml
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np

from naturalv2.utils import check_trial
from naturalv2.evals.experiment import Experiment

def find_valid_ncts(valid_nct_path, cfg):
    nct_list = []
    for filename in os.listdir(cfg.data_path + "/nct_reports"):
        if filename.endswith('.json'):
            nct_id = filename[:-5] # ends with .json
            trial = instantiate(cfg.eval, nct_id=nct_id)
            if check_trial(trial):
                nct_list.append(nct_id)
                with open(valid_nct_path, 'a') as f:
                    f.write(f"{nct_id}\n")

def find_indication_ncts(indication_nct_path, nct_list, cfg):
    retro_trials = []
    for nct_id in nct_list:
        trial = instantiate(cfg.eval, nct_id=nct_id)
        # check if {indication} is mentioned in the trial's list of conditions or keywords
        if any(cfg.indication.lower() in word.lower() for word in trial.conditions + trial.keywords):
            retro_trials.append((nct_id, trial.results_first_posted))
            with open(indication_nct_path, 'a') as f:
                f.write(f"{nct_id}\n")

class RetroStudy:
    def __init__(self, retro_trials, cfg):
        # order retro_trials by completion date and split into train/val according to train_ratio
        retro_trials.sort(key=lambda x: x[1])
        train_size = int(len(retro_trials) * cfg.train_ratio) 
        self.train_trials, self.val_trials = retro_trials[:train_size], retro_trials[train_size:]

        self.indication = cfg.indication
        self.train_ratio = cfg.train_ratio
        self.create_exp(cfg.eval.data_path)

        print(f"Study created for {self.indication} with {len(self.train_trials)} training and {len(self.val_trials)} validation experiments!")

    def to_yaml(self, filename):
        with open(filename, "w") as file:
            yaml.dump(self.__dict__, file)

    def create_exp(self, data_path):
        train_exp = [Experiment(nct_id, data_path, train=True) for nct_id in self.train_trials]
        val_exp = [Experiment(nct_id, data_path, train=False) for nct_id in self.val_trials]


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    # find nct_ids of valid retrospective trials
    valid_nct_path = os.path.join(cfg.data_path, 'nct_reports/valid_binary_nct_ids.txt')
    if not os.path.exists(valid_nct_path):
        find_valid_ncts(valid_nct_path, cfg)
    with open(valid_nct_path, 'r') as f:
        nct_list = [line.strip() for line in f.readlines()]

    # find nct_ids of retrospective trials related to indication
    indication_nct_path = os.path.join(cfg.data_path, f'nct_reports/valid_binary_{cfg.indication}_nct_ids.txt')
    if not os.path.exists(indication_nct_path):
        find_indication_ncts(indication_nct_path, nct_list, cfg)
    with open(indication_nct_path, 'r') as f:
        retro_trials = [line.strip() for line in f.readlines()]
    
    study = RetroStudy(retro_trials, cfg)
    study.to_yaml(os.path.join(cfg.save_path, cfg.indication + "_study.yaml"))

if __name__ == "__main__":
    main()