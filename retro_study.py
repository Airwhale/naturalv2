import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import numpy as np

from naturalv2.utils import check_trial

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    data_path = cfg.trial.source.data_path
    if not os.path.exists(os.path.join(data_path, 'valid_nct_ids.txt')):
        nct_list = []
        for filename in os.listdir(data_path):
            if filename.endswith('.json'):
                nct_id = filename[:-5] # ends with .json
                trial = ClinicalTrial(data_path, nct_id)
                if check_trial(trial):
                    nct_list.append(nct_id)
                    with open(os.path.join(data_path, 'valid_nct_ids.txt'), 'a') as f:
                        f.write(f"{nct_id}\n")

    with open(os.path.join(data_path, 'valid_nct_ids.txt'), 'r') as f:
        nct_list = [line.strip() for line in f.readlines()]

    indication = cfg.indication
    if not os.path.exists(os.path.join(data_path, f'valid_{indication}_nct_ids.txt')):
        retro_trials = []
        for nct_id in nct_list:
            trial = instantiate(cfg.trial.source, nctid=nct_id)
            # check if {indication} is mentioned in the trial's list of conditions or keywords
            if any(indication.lower() in word.lower() for word in trial.conditions + trial.keywords):
                retro_trials.append(nct_id)
                with open(os.path.join(data_path, f'valid_{indication}_nct_ids.txt'), 'a') as f:
                    f.write(f"{nct_id}\n")
            
    with open(os.path.join(data_path, f'valid_{indication}_nct_ids.txt'), 'r') as f:
        retro_trials = [line.strip() for line in f.readlines()]

    import pdb; pdb.set_trace()

    
    # download subreddits with relevant information and curate
    # download pubmed case studies with relevant information and curate

    llm = instantiate(cfg.model)
    for trial in trials:
        reddit_set = instantiate(cfg.reddit.source, trial=trial, download=True)
        reddit_curated = reddit_set.curate_data(date_filter=cfg.filter_by_date)

        pubmed_set = instantiate(cfg.pubmed.source, trial=trial, download=True)
        pubmed_curated = pubmed_set.curate_data(date_filter=cfg.filter_by_date)

        extracted_variables = run_extract(reddit_curated, pubmed_curated, llm)
        pred_ate = estimate_ate(extracted_variables, cfg.ate_estimator)
        ground_truth = trial.ground_truth

        error_magnitude = np.abs(pred_ate - ground_truth)
        error_direction = (pred_ate * ground_truth) < 0


if __name__ == "__main__":
    main()