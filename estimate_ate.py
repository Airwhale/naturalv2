import json
import os

import hydra
import nest_asyncio
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from naturalv2.evals.svt import SvT
from naturalv2.utils import (
    enum_to_dcts,
    enumerate_strings,
    get_sample_text,
    qa_interleaved_enum,
)


def extract_covariates(input_df, experiment, model, save_path, extract_type):
    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=0)
    model.system_prompt = experiment.get_prompt(extract_type)
    model.human_template = "\n## Input \n>{report}"

    llm_samples_df = pd.DataFrame()
    llm_inputs = []
    for _, row in tqdm(input_df.iterrows()):
        report = row["report"]
        llm_inputs.append({"report": report})
        if len(llm_inputs) >= model.batch_size or len(input_df) == len(
            llm_samples_df
        ) + len(llm_inputs):
            llm_out_dicts = model.get_outputs(model.system_prompt, llm_inputs)
            llm_out_dicts = [json.loads(text) for text in llm_out_dicts]
            dict_to_save = [
                {**llm_out_dicts[j], **{"report": llm_inputs[j]["report"]}}
                for j in range(len(llm_inputs))
            ]
            df_to_save = pd.DataFrame.from_dict(dict_to_save)
            llm_samples_df = pd.concat([llm_samples_df, df_to_save], ignore_index=True)
            if (
                extract_type == "imputations"
            ):  # TODO later: Remove to use only new extractions - shouldn't change results much.
                input_df.update(llm_samples_df, overwrite=False)
                llm_samples_df = input_df.copy().iloc[: len(llm_samples_df)]
            if extract_type != "ty_filter":
                llm_samples_df = experiment.discretize(
                    llm_samples_df, hard_filter=False, inf=False
                )
            llm_samples_df.to_csv(save_path)
            llm_inputs = []

    llm_samples_df.to_csv(save_path)
    return llm_samples_df


def filter_by_ty(samples_df, experiment):
    return experiment.hard_filter_ty(samples_df)


def filter_by_inclusion(samples_df, experiment):
    return experiment.discretize(samples_df, hard_filter=True, inf=False)
    # samples_df = samples_df.map(lambda x: np.nan if x in ["Unknown", "unknown"] else x)


def extract_conditionals(input_df, experiment, model, save_path, inclusion=False):
    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=0)
    input_df = experiment.discretize(input_df, hard_filter=False, inf=True)
    model.system_prompt = experiment.get_prompt("conditionals")
    llm_probs_df = pd.DataFrame()
    to_enum = ["inclusion"] if inclusion else ["treatment"] + experiment.outcome_names
    options = enumerate_strings(experiment.get_options(to_enum))
    interleaved_options = qa_interleaved_enum(
        experiment.get_question_prompt(to_enum),
        experiment.get_options(to_enum),
        options,
        to_enum,
    )
    idx_to_feat = enum_to_dcts(options, to_enum)
    idx_to_feat = [experiment.transform_samples(dct) for dct in idx_to_feat]
    llm_inputs, rows = [], []

    for _, row in tqdm(input_df.iterrows()):
        X = row["report"]
        if not inclusion:
            # get corresponding row from input_df
            sample_row = input_df.loc[input_df["report"] == X]
            if len(sample_row) == 0:
                continue
            sample_row = sample_row[experiment.covariate_names]
            sample_row = sample_row.to_dict("records")[0]
            sample_text = get_sample_text(sample_row, experiment)
            X += sample_text

        llm_inputs.append(X)
        cols = (
            experiment.covariate_names
            + experiment.outcome_names
            + ["treatment", "report"]
        )
        rows.append(row[cols])
        if len(llm_inputs) >= model.batch_size or len(input_df) == len(
            llm_probs_df
        ) + len(llm_inputs):
            post_probs, sample_indices, _ = model.compute_input_probs(
                llm_inputs, interleaved_options
            )
            dict_to_save = [
                {
                    **rows[j].to_dict(),
                    **idx_to_feat[sample_indices[j]],
                    **{"probs": post_probs[j]},
                }
                for j in range(len(llm_inputs))
            ]
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

    cheap_model = instantiate(cfg.cheap_model, response_format={"type": "json_object"})
    sample_model = instantiate(
        cfg.sample_model, response_format={"type": "json_object"}
    )
    nest_asyncio.apply()

    data_flow = {}

    curated_df = pd.read_csv(experiment.curated_data_path, index_col=0).head(10)
    data_flow["curated"] = len(curated_df)
    print(f"Initial number of curated reports: {len(curated_df)} reports.")

    # filter reports that do not contain t,y info
    ty_path = os.path.join(
        cfg.save_path, f"{experiment.nct_id}/{cheap_model.model_name}_ty_samples.csv"
    )
    ty_samples = extract_covariates(
        curated_df, experiment, cheap_model, ty_path, "ty_filter"
    )
    ty_filtered_df = filter_by_ty(ty_samples, experiment)
    data_flow["ty_filtered"] = len(ty_filtered_df)
    print(f"After treatment-outcome filter: {len(ty_filtered_df)} reports.")

    # extract samples from reports, allowing LLM to output "unknown" for missing info
    knowns_path = os.path.join(
        cfg.save_path,
        f"{experiment.nct_id}/{sample_model.model_name}_samples_knowns.csv",
    )
    samples_with_unknown = extract_covariates(
        ty_filtered_df, experiment, sample_model, knowns_path, "knowns"
    )

    # filter reports known to violate inclusion criteria
    inclusion_filtered = filter_by_inclusion(samples_with_unknown, experiment)
    data_flow["inclusion_filtered"] = len(inclusion_filtered)
    print(f"After inclusion filter: {len(inclusion_filtered)} reports.")

    # impute samples from reports, imputing missing info
    imputed_path = os.path.join(
        cfg.save_path,
        f"{experiment.nct_id}/{sample_model.model_name}_samples_imputed.csv",
    )
    imputed_samples = extract_covariates(
        inclusion_filtered, experiment, sample_model, imputed_path, "imputations"
    )
    # drop rows with missing covariates even after imputation
    imputed_samples = imputed_samples.dropna(
        subset=experiment.covariate_names
    ).reset_index(drop=True)
    data_flow["final"] = len(imputed_samples)
    print(f"Final: {len(imputed_samples)} reports.")

    probs_model = instantiate(cfg.probs_model)
    if cfg.load_model:
        probs_model.load_model()
    probs_model_name = probs_model.model_name.replace(
        "/", "_"
    )  # vllm-supported models often have a "/"

    # extract conditionals of the form P(T, Y | X, R)
    conditionals_path = os.path.join(
        cfg.save_path, f"{experiment.nct_id}/{probs_model_name}_conditionals.csv"
    )
    conditionals = extract_conditionals(
        imputed_samples, experiment, probs_model, conditionals_path
    )

    # extract inclusion probabilities of the form P(X in I | R)
    inclusion_path = os.path.join(
        cfg.save_path, f"{experiment.nct_id}/{probs_model_name}_inclusion_probs.csv"
    )
    inclusion_probs = extract_conditionals(
        imputed_samples, experiment, probs_model, inclusion_path, inclusion=True
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
                        "outcome": outcome,
                        "treatments": f"{treat2}-{treat1}",
                        "pred_ate": pred_ate,
                    }
                    print("Predicted ATE: ", pred_ate)
                    if experiment.split != "test":
                        effect_idx = experiment.outcome_treatment.index(
                            (outcome, (treat1, treat2))
                        )
                        true_ate = experiment.effect_sizes[effect_idx]
                        error = abs(pred_ate - true_ate)
                        results.update({"true_ate": true_ate, "abs_error": error})
                        print("True ATE: ", true_ate)
                        print("Absolute Error: ", error)
                    results.update(data_flow)
                    result_dicts.append(results)

    # TODO later: compute other evaluation metrics, e.g. sensitivity, balance

    result_df = pd.DataFrame(result_dicts)
    result_df.to_csv(
        os.path.join(cfg.save_path, f"{experiment.nct_id}/ate_results.csv")
    )


if __name__ == "__main__":
    main()
