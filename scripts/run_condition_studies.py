import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from hydra import initialize, compose

from create_study import run_study_and_get_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONDITION_LISTS = [
    ["Animal Diseases"],
    ["Cardiovascular Diseases"],
    ["Congenital, Hereditary, and Neonatal Diseases and Abnormalities"],
    ["Digestive System Diseases"],
    ["Disorders of Environmental Origin"],
    ["Endocrine System Diseases"],
    ["Eye Diseases"],
    ["Hemic and Lymphatic Diseases"],
    ["Immune System Diseases"],
    ["Infections"],
    ["Musculoskeletal Diseases"],
    ["Neoplasms"],
    ["Nervous System Diseases"],
    ["Nutritional and Metabolic Diseases"],
    ["Occupational Diseases"],
    ["Otorhinolaryngologic Diseases"],
    ["Pathological Conditions, Signs and Symptoms"],
    ["Respiratory Tract Diseases"],
    ["Skin and Connective Tissue Diseases"],
    ["Stomatognathic Diseases"],
    ["Urogenital Diseases"],
    ["Wounds and Injuries"],
]


def run_study(conditions: list[str], args: argparse.Namespace) -> dict[str, Any]:
    
    config_dir = os.path.dirname(args.config_path)
    config_name = os.path.basename(args.config_path)
    if config_name.endswith('.yaml'):
        config_name = config_name[:-5]
    with initialize(config_path=config_dir or None, version_base="1.2"):
        cfg = compose(config_name=config_name)
    
    cfg.conditions = conditions
    cfg.save_path = args.output_dir
   
    logger.info(f"Running study for: {conditions}")
    stats = run_study_and_get_stats(cfg)
    return stats


def count_unique_ncts(studies_dir: str) -> dict[str, int]:
    train_ncts = set()
    val_ncts = set()
    test_ncts = set()

    for yaml_file in Path(studies_dir).glob("**/*.yaml"):
        with open(yaml_file, "r") as f:
            study_data = yaml.safe_load(f)

            # Extract NCT IDs from train trials
            for trial in study_data["train_trials"]:
                train_ncts.update(trial.keys())
            
            # Extract NCT IDs from val trials
            for trial in study_data["val_trials"]:
                val_ncts.update(trial.keys())
            
            # Extract NCT IDs from test trials
            for trial in study_data["test_trials"]:
                test_ncts.update(trial.keys())

    # Count total labels across unique trials
    total_train_labels = 0
    total_val_labels = 0
    total_test_labels = 0
    for yaml_file in Path(studies_dir).glob("**/*.yaml"):
        with open(yaml_file, "r") as f:
            study_data = yaml.safe_load(f)
            
            # Sum up the first element (number of labels) for each unique train trial
            for trial in study_data["train_trials"]:
                for nct_id in trial.keys():
                    if nct_id in train_ncts:
                        total_train_labels += trial[nct_id][0]
            
            # Sum up the first element (number of labels) for each unique val trial
            for trial in study_data["val_trials"]:
                for nct_id in trial.keys():
                    if nct_id in val_ncts:
                        total_val_labels += trial[nct_id][0]
            
            # Sum up the first element (number of labels) for each unique test trial
            for trial in study_data["test_trials"]:
                for nct_id in trial.keys():
                    if nct_id in test_ncts:
                        total_test_labels += trial[nct_id][0]

    return {
        "train_trials": len(train_ncts),
        "val_trials": len(val_ncts),
        "test_trials": len(test_ncts),
        "train_labels": total_train_labels,
        "val_labels": total_val_labels,
        "test_labels": total_test_labels
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run create_study with different conditions and record stats"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="../conf/config.yaml",
        help="Path to the config.yaml file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save the output files"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run all the studies and collect results
    results = []
    for conditions in CONDITION_LISTS:
        results.append(run_study(conditions, args))

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Save to CSV
    csv_path = os.path.join(args.output_dir, "create_study_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    # logger.info(
    #     f"{count_unique_ncts(os.path.join(args.output_dir, 'studies'))} unique trials and labels covered."
    # )
