import os
import pandas as pd
import json

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from naturalv2.utils import check_trial, check_nonplacebo
from naturalv2.sources.clinicaltrials import ClinicalTrial

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    # llm = instantiate(cfg.model)
    trial = instantiate(cfg.trial.source)
    data_path = trial.data_path
    # reddit_set = instantiate(cfg.reddit.source, download=True)
    # pubmed_set = instantiate(cfg.pubmed.source, download=True)
    

    # import pdb; pdb.set_trace()
    # assert check_trial(trial)
    condition = "migraine"

    total, randomized, non_placebo, no_healthy_vol, migraine = 0, 0, 0, 0, 0

    
    migraine_list = []
    with open(os.path.join(data_path, 'valid_nct_ids.txt'), 'r') as f:
        nct_list = [line.strip() for line in f.readlines()]

    for nct_id in nct_list:
        trial = ClinicalTrial(data_path, nct_id)
        if any(condition.lower() in word.lower() for word in trial.conditions + trial.keywords):
            migraine += 1
            migraine_list.append(nct_id)
            with open(os.path.join(data_path, f'valid_{condition}_nct_ids.txt'), 'a') as f:
                f.write(f"{nct_id}\n")


    # nct_list = []
    # for filename in os.listdir(data_path):
    #     if filename.endswith('.json'):
    #         nct_id = filename[:-5] # ends with .json
    #         trial = ClinicalTrial(data_path, nct_id)
    #         total += 1
            
    #         if trial.alloc == "RANDOMIZED":
    #             randomized += 1
                
    #             nonplacebo_interventions = [i.title for i in trial.interventions if check_nonplacebo(i.title)]
    #             if len(nonplacebo_interventions) >= 2:
    #                 non_placebo += 1

    #                 if trial.inclusion_criteria.healthy_volunteers != "" and not trial.inclusion_criteria.healthy_volunteers:
    #                     no_healthy_vol += 1
    #                     nct_list.append(nct_id)
    #                     with open(os.path.join(data_path, 'valid_nct_ids.txt'), 'a') as f:
    #                         f.write(f"{nct_id}\n")
                        
    #                     if any(condition.lower() in word.lower() for word in trial.conditions + trial.keywords):
    #                         diabetes += 1
    #                         diabetes_list.append(nct_id)
    #                         with open(os.path.join(data_path, 'valid_diabetes_nct_ids.txt'), 'a') as f:
    #                             f.write(f"{nct_id}\n")
                        



    # print("Total: ", total)
    # print("Randomized: ", randomized)
    # print("Non-placebo: ", non_placebo)
    # print("No healthy volunteers: ", no_healthy_vol)
    print(f"{condition}-related: ", migraine)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()