import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    trial = instantiate(cfg.eval)
    llm = instantiate(cfg.model)

    reddit = instantiate(cfg.reddit.source, trial=trial, download=True)
    reddit_curated = reddit.curate_data(date_filter=cfg.filter_by_date)

    pubmed = instantiate(cfg.pubmed.source, trial=trial, download=True)
    pubmed_curated = pubmed.curate_data(date_filter=cfg.filter_by_date)

    # TODO: create a save an Experiment object (yaml?) similar to datasets in NATURAL


if __name__ == "__main__":
    main()
