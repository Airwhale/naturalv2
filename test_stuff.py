import pandas as pd
import json

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from naturalv2.utils import check_trial, check_nonplacebo

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    llm = instantiate(cfg.model)
    trial = instantiate(cfg.trial.source)
    reddit_set = instantiate(cfg.reddit.source, download=True)
    pubmed_set = instantiate(cfg.pubmed.source, download=True)
    

    import pdb; pdb.set_trace()
    assert check_trial(trial)

    total, randomized, non_placebo, no_healthy_vol = 0, 0, 0, 0

    nct_list = []
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            nct_id = filename[:-5]
            trial = ClinicalTrial(data_path, nct_id)
            total += 1
            
            if trial.alloc == "RANDOMIZED":
                randomized += 1
                
                nonplacebo_interventions = [i.title for i in trial.interventions if check_nonplacebo(i.title)]
                if len(nonplacebo_interventions) >= 2:
                    non_placebo += 1

                    if trial.inclusion_criteria.healthy_volunteers != "" and not trial.inclusion_criteria.healthy_volunteers:
                        no_healthy_vol += 1
                        
                        nct_list.append(nct_id)


    print("Total: ", total)
    print("Randomized: ", randomized)
    print("Non-placebo: ", non_placebo)
    print("No healthy volunteers: ", no_healthy_vol)


if __name__ == "__main__":
    main()