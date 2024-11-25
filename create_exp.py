import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from naturalv2.utils import check_trial

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    trial = instantiate(cfg.trial.source)
    assert check_trial(trial)

    llm = instantiate(cfg.model)
    
    reddit_set = instantiate(cfg.reddit.source, download=True)
    reddit_curated = reddit_set.curate_data(date_filter=cfg.filter_by_date)
    
    import pdb; pdb.set_trace()
    pubmed_set = instantiate(cfg.pubmed.source, download=True)


if __name__ == "__main__":
    main()