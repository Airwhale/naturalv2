import logging
import os

import hydra
import nest_asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from naturalv2.evals.experiment import Experiment

load_dotenv(".env")


def weight_by_inclusion(ites: np.ndarray, inclusion_probs: pd.DataFrame) -> np.ndarray:
    """Weight ITEs by inclusion probabilities."""
    # ites has shape [num_treatments, num_datapoints]
    probs = inclusion_probs.apply(
        lambda row: [float(prob) for prob in row["inclusion_probs"][1:-1].split()][1], axis=1
    ).to_numpy()
    return np.average(ites, axis=1, weights=probs)


def calculate_treatment_effects(
    experiment: Experiment,
    outcome: str,
    estimator,
    extractions: pd.DataFrame,
    data_flow: dict[str, int],
) -> list[dict]:
    """Calculate treatment effects for all outcome-treatment pairs."""
    result_dicts = []

    if hasattr(estimator, "estimator_type"):
        all_ites = estimator.get_ites(extractions, outcome)
    else:
        all_ites = estimator.get_ites(extractions)
    weighted_effects = weight_by_inclusion(
        all_ites, extractions
    )  # len: num_treatments

    for i, treat1 in enumerate(experiment.treatment_names):
        for j, treat2 in enumerate(experiment.treatment_names):
            if i < j:
                pred_ate = weighted_effects[j] - weighted_effects[i]
                results = {
                    "estimator": estimator.__class__.__name__,
                    "outcome": outcome,
                    "treatments": f"{treat2}-{treat1}",
                    "pred_ate": pred_ate,
                }
                logging.info(f"Predicted ATE: {pred_ate}")
                if experiment.split != "test":
                    effect_idx = experiment.outcome_treatment.index(
                        (outcome, (treat1, treat2))
                    )
                    true_ate = experiment.effect_sizes[effect_idx]
                    error = abs(pred_ate - true_ate)
                    results.update({"true_ate": true_ate, "abs_error": error})
                    logging.info(f"True ATE: {true_ate}")
                    logging.info(f"Absolute Error: {error}")
                results.update(data_flow)
                result_dicts.append(results)

    return result_dicts


def save_results(results: list[dict], save_path: str, nct_id: str) -> None:
    """Save results to CSV file."""
    result_df = pd.DataFrame(results)
    results_path = os.path.join(save_path, "results", f"{nct_id}/ate_results.csv")

    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path, index_col=0)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    result_df.to_csv(results_path)


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main function to estimate average treatment effects."""
    exp_file = os.path.join(cfg.save_path, "experiments", f"{cfg.eval.nct_id}.yaml")
    experiment = Experiment.from_yaml(exp_file)
    os.makedirs(
        os.path.join(cfg.save_path, "results", f"{experiment.nct_id}"), exist_ok=True
    )
    source_name = cfg.sources[0]
    outcome = cfg.outcome or experiment.outcome_names[0]

    pipeline = instantiate(
        cfg.pipeline,
        experiment=experiment,
        source_name=source_name,
        estimator_type=cfg.estimator._target_.split(".")[-1],
        outcome=outcome,
        save_path=cfg.save_path,
    )
    for stage_config in cfg.pipeline_stages:
        stage = instantiate(stage_config)
        pipeline.add_stage(stage)
    
    nest_asyncio.apply()

    # Load curated data for the first source in {cfg.sources}
    curated_df = pd.concat(
        [
            pd.read_csv(path, index_col=0)
            for path in experiment.source_paths[source_name]
        ],
        ignore_index=True,
    )
    # TODO: remove subsampling after testing
    curated_df = curated_df.sample(frac=0.05, random_state=cfg.seed, ignore_index=True)
    pipeline.data_flow["initial_curated"] = len(curated_df)
    logging.info(f"Initial number of curated reports: {len(curated_df)} reports.")

    extractions = pipeline.run(curated_df)

    # Calculate and save treatment effects
    estimator = instantiate(cfg.estimator, experiment=experiment)
    results = calculate_treatment_effects(
        experiment, pipeline.config.outcome, estimator, extractions, pipeline.data_flow
    )

    save_results(results, cfg.save_path, experiment.nct_id)


if __name__ == "__main__":
    main()
