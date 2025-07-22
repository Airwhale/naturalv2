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
