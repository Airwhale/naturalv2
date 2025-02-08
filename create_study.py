import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np

from naturalv2.utils import check_trial

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    # find nct_ids of valid retrospective trials
    valid_nct_path = os.path.join(cfg.data_path, 'nct_reports/valid_binary_nct_ids.txt')
    if not os.path.exists(valid_nct_path):
        nct_list = []
        for filename in os.listdir(cfg.data_path + "/nct_reports"):
            if filename.endswith('.json'):
                nct_id = filename[:-5] # ends with .json
                trial = instantiate(cfg.eval, nctid=nct_id)
                if check_trial(trial):
                    nct_list.append(nct_id)
                    with open(valid_nct_path, 'a') as f:
                        f.write(f"{nct_id}\n")
    
    with open(valid_nct_path, 'r') as f:
        nct_list = [line.strip() for line in f.readlines()]

    # find nct_ids of retrospective trials related to indication
    indication_nct_path = os.path.join(cfg.data_path, f'nct_reports/valid_binary_{cfg.indication}_nct_ids.txt')
    if not os.path.exists(indication_nct_path):
        retro_trials = []
        for nct_id in nct_list:
            trial = instantiate(cfg.eval, nctid=nct_id)
            # check if {indication} is mentioned in the trial's list of conditions or keywords
            if any(cfg.indication.lower() in word.lower() for word in trial.conditions + trial.keywords):
                retro_trials.append((nct_id, trial.results_first_posted))
                with open(indication_nct_path, 'a') as f:
                    f.write(f"{nct_id}\n")

    with open(indication_nct_path, 'r') as f:
        retro_trials = [line.strip() for line in f.readlines()]
    
    # order retro_trials by completion date
    retro_trials.sort(key=lambda x: x[1])
    # split into train/val according to train_ratio
    train_size = int(len(retro_trials) * cfg.train_ratio) 
    train_trials, val_trials = retro_trials[:train_size], retro_trials[train_size:]

    print(f"{len(train_trials)} training and {len(val_trials)} validation trials found!")

    # TODO: create a Study object with train and validation experiments where each valid binary outcome is a separate experiment

if __name__ == "__main__":
    main()