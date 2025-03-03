import asyncio
import json
import logging
import os
import sys
from typing import Literal, Optional

import hydra
import nest_asyncio
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import BaseModel
from scipy.special import softmax
from tqdm import tqdm

from naturalv2.evals.svt import SvT
from naturalv2.models.lm import LM
from naturalv2.utils import (
    ImputationsResponse,
    KnownsResponse,
    TYFilterResponse,
    enum_to_dcts,
    enumerate_strings,
    get_sample_text,
    qa_interleaved_enum,
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


async def extract_covariates(
    input_df: pd.DataFrame,
    experiment: SvT,
    model_cfg: DictConfig,
    save_path: str,
    extract_type: Literal["ty_filter", "knowns", "imputations"],
    batch_size: int = 1,
    response_format: Optional[type[BaseModel]] = None,
):
    if extract_type == "ty_filter":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}",
            f"{model_cfg.model.replace('/', '-')}_ty_samples.csv",
        )
    elif extract_type == "knowns":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}/{model_cfg.model.replace('/', '-')}_samples_knowns.csv",
        )
    elif extract_type == "imputations":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}/{model_cfg.model.replace('/', '-')}_samples_imputed.csv",
        )
    else:
        raise ValueError(
            f"Invalid extract_type. Expected one of ['ty_filter', 'knowns', 'imputations'], "
            f"but got {extract_type}."
        )

    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=0)

    model = LM(**model_cfg)

    system_msg = {"role": "system", "content": experiment.get_prompt(extract_type)}
    human_template = "\n## Input \n>{report}"

    out_dicts = []

    for start in tqdm(range(0, len(input_df), batch_size)):
        batch_df = input_df.iloc[start : start + batch_size]

        reports = batch_df["report"].tolist()
        lm_responses = await asyncio.gather(  # guaranteed to be in order
            *(
                model.apredict(
                    messages=[
                        system_msg,
                        {
                            "role": "user",
                            "content": human_template.format(report=report),
                        },
                    ],
                    response_format=response_format,
                    # extra_body={"guided_json": response_format.model_json_schema()},
                )
                for report in reports
            )
        )
        parsed_lm_responses: list[dict] = [
            json.loads(text) for response in lm_responses for text in response
        ]

        out_dicts.extend(
            [
                {**parsed_lm_responses[j], **{"report": reports[j]}}
                for j in range(len(batch_df))
            ]
        )

    llm_samples_df = pd.DataFrame.from_dict(out_dicts)
    if (
        extract_type == "imputations"
    ):  # TODO later: Remove to use only new extractions - shouldn't change results much.
        input_df.update(llm_samples_df, overwrite=False)
        llm_samples_df = input_df.copy()

    if extract_type != "ty_filter":
        llm_samples_df = experiment.discretize(
            llm_samples_df, hard_filter=False, inf=False
        )

    llm_samples_df.to_csv(save_path)
    return llm_samples_df


def filter_by_ty(samples_df, experiment):
    return experiment.hard_filter_ty(samples_df)


def filter_by_inclusion(samples_df, experiment):
    return experiment.discretize(samples_df, hard_filter=True, inf=False)
    # samples_df = samples_df.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)


def extract_conditionals(
    input_df: pd.DataFrame,
    experiment: SvT,
    model_cfg: DictConfig,
    save_path: str,
    extract_type: Literal["ty_given_x", "y_given_tx", "inclusion", None],
    length_norm: bool = False,
    batch_size: int = 1,
):
    # TODO for NOI: specify different possible conditionals to extract: (X \in I | X), (T,Y|X), (Y|T,X)
    if extract_type == "ty_given_x":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}",
            f"{model_cfg.model.replace('/', '-')}_ty_given_x.csv",
        )
        to_enum = ["treatment"] + experiment.outcome_names
    elif extract_type == "y_given_tx":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}",
            f"{model_cfg.model.replace('/', '-')}_y_given_tx.csv",
        )
        to_enum = experiment.outcome_names
    elif extract_type == "inclusion":
        save_path = os.path.join(
            save_path,
            f"{experiment.nct_id}",
            f"{model_cfg.model.replace('/', '-')}_inclusion_probs.csv",
        )
        to_enum = ["inclusion"]
    else:
        return input_df

    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=0)

    # validate model_cfg
    assert model_cfg.get("model_type") == "text", "Model type must be 'text'."
    assert model_cfg.get("prompt_logprobs") == 0, "Prompt logprobs must be 0."

    model = LM(**model_cfg)

    input_df = experiment.discretize(input_df, hard_filter=False, inf=True)

    system_prompt = experiment.get_prompt("conditionals")

    options = enumerate_strings(experiment.get_options(to_enum))
    interleaved_options = qa_interleaved_enum(
        experiment.get_question_prompt(to_enum),
        experiment.get_options(to_enum),
        options,
        to_enum,
    )
    idx_to_feat = enum_to_dcts(options, to_enum)
    idx_to_feat = [experiment.transform_samples(dct) for dct in idx_to_feat]

    llm_probs_df = pd.DataFrame()

    for start in tqdm(range(0, len(input_df), batch_size)):
        batch_df = input_df.iloc[start : start + batch_size].reset_index(drop=True)

        reports = batch_df["report"].tolist()
        if extract_type != "inclusion":
            for idx, report in enumerate(reports):
                row = input_df.loc[input_df["report"] == report]
                if len(row) == 0:
                    continue
                to_sample = experiment.covariate_names if extract_type == "ty_given_x" else experiment.covariate_names + ["treatment"]
                row = row[to_sample].to_dict("records")[0]
                sample_text = get_sample_text(row, experiment)
                reports[idx] += sample_text

        reports_repeated = [
            report for report in reports for _ in range(len(interleaved_options))
        ]
        options_repeated = interleaved_options * len(reports)
        llm_inputs = [
            report + option
            for report, option in zip(reports_repeated, options_repeated)
        ]

        cols = (
            experiment.covariate_names
            + experiment.outcome_names
            + ["treatment", "report"]
        )
        rows = batch_df[cols]

        lm_responses = [
            model.predict(prompt=system_prompt + "\n\n" + llm_input)
            for llm_input in llm_inputs
        ]

        logprobs = []
        for lm_response in lm_responses:
            logprob = sum(lm_response[0]["prompt_logprobs"])
            if length_norm:
                logprob = logprob / len(lm_response[0]["prompt_tokens"])
            logprobs.append(logprob)

        probs = softmax(
            np.array(logprobs).reshape((len(reports), len(interleaved_options))),
            axis=1,
        )
        sample_indices = [np.random.choice(len(prob), p=prob) for prob in probs]

        dict_to_save = [
            {
                **rows.iloc[j].to_dict(),
                **idx_to_feat[sample_indices[j]],
                **{"probs": probs[j]},
            }
            for j in range(len(reports))
        ]

        # TODO [fcogidi]: avoid saving to disk at every iteration?
        df_to_save = pd.DataFrame.from_dict(dict_to_save)
        llm_probs_df = pd.concat([llm_probs_df, df_to_save], ignore_index=True)
        llm_probs_df.to_csv(save_path)
        llm_inputs, rows = [], []

    llm_probs_df.to_csv(save_path)
    return pd.read_csv(save_path, index_col=0)


def weight_by_inclusion(ites, inclusion_probs):
    # ites has shape [num_treatments, num_datapoints]
    probs = inclusion_probs.apply(
        lambda row: [float(prob) for prob in row["probs"][1:-1].split()][1], axis=1
    ).to_numpy()
    return np.average(ites, axis=1, weights=probs)


@hydra.main(config_path="conf/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    experiment = SvT(path_to_main=cfg.user.path_to_main)
    os.makedirs(os.path.join(cfg.save_path, f"{experiment.nct_id}"), exist_ok=True)

    nest_asyncio.apply()

    data_flow = {}

    curated_df = pd.read_csv(experiment.curated_data_path, index_col=0).head(10)
    data_flow["curated"] = len(curated_df)
    logger.info(f"Initial number of curated reports: {len(curated_df)} reports.")

    # filter reports that do not contain t,y info
    ty_samples = asyncio.run(
        extract_covariates(
            curated_df,
            experiment,
            cfg.cheap_model,
            cfg.save_path,
            "ty_filter",
            response_format=TYFilterResponse,
        )
    )
    ty_filtered_df = filter_by_ty(ty_samples, experiment)
    data_flow["ty_filtered"] = len(ty_filtered_df)
    logger.info(f"After treatment-outcome filter: {len(ty_filtered_df)} reports.")

    # extract samples from reports, allowing LLM to output "unknown" for missing info
    samples_with_unknown = asyncio.run(
        extract_covariates(
            ty_filtered_df,
            experiment,
            cfg.sample_model,
            cfg.save_path,
            "knowns",
            response_format=KnownsResponse,
        )
    )

    # filter reports known to violate inclusion criteria
    inclusion_filtered = filter_by_inclusion(samples_with_unknown, experiment)
    data_flow["inclusion_filtered"] = len(inclusion_filtered)
    logger.info(f"After inclusion filter: {len(inclusion_filtered)} reports.")

    # impute samples from reports, imputing missing info
    imputed_samples = asyncio.run(
        extract_covariates(
            inclusion_filtered,
            experiment,
            cfg.sample_model,
            cfg.save_path,
            "imputations",
            response_format=ImputationsResponse,
        )
    )

    # drop rows with missing covariates even after imputation
    imputed_samples = imputed_samples.dropna(
        subset=experiment.covariate_names
    ).reset_index(drop=True)
    data_flow["final"] = len(imputed_samples)
    logger.info(f"Final: {len(imputed_samples)} reports.")

    # extract conditionals depending on the estimator type
    estimator_type = cfg.estimator._target_.split(".")[-1]
    extract_type = {
        "NaturalIPW": "ty_given_x",
        "NaturalOI": "y_given_tx",
        "NaturalMC": None,
    }[estimator_type]
    conditionals = extract_conditionals(
        imputed_samples, experiment, cfg.probs_model, cfg.save_path, extract_type=extract_type
    )

    # extract inclusion probabilities of the form P(X in I | R)
    inclusion_probs = extract_conditionals(
        imputed_samples, experiment, cfg.probs_model, cfg.save_path, extract_type="inclusion"
    )

    estimator = instantiate(cfg.estimator, experiment=experiment)
    result_dicts = []
    for outcome in experiment.outcome_names:
        all_ites = estimator.get_ites(conditionals, outcome)
        weighted_effects = weight_by_inclusion(
            all_ites, inclusion_probs
        )  # len: num_treatments

        for i, treat1 in enumerate(experiment.treatment_names):
            for j, treat2 in enumerate(experiment.treatment_names):
                if i < j:
                    pred_ate = weighted_effects[j] - weighted_effects[i]
                    results = {
                        "estimator": cfg.estimator._target_.split(".")[-1],
                        "outcome": outcome,
                        "treatments": f"{treat2}-{treat1}",
                        "pred_ate": pred_ate,
                    }
                    logger.info(f"Predicted ATE: {pred_ate}")
                    if experiment.split != "test":
                        effect_idx = experiment.outcome_treatment.index(
                            (outcome, (treat1, treat2))
                        )
                        true_ate = experiment.effect_sizes[effect_idx]
                        error = abs(pred_ate - true_ate)
                        results.update({"true_ate": true_ate, "abs_error": error})
                        logger.info(f"True ATE: {true_ate}")
                        logger.info(f"Absolute Error: {error}")
                    results.update(data_flow)
                    result_dicts.append(results)

    # TODO later: compute other evaluation metrics, e.g. sensitivity, balance

    result_df = pd.DataFrame(result_dicts)
    results_path = os.path.join(cfg.save_path, f"{experiment.nct_id}/ate_results.csv")
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path, index_col=0)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    result_df.to_csv(results_path)


if __name__ == "__main__":
    main()
