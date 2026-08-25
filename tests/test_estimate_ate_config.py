from pathlib import Path

from hydra import compose, initialize_config_dir

import naturalv2.hydra_setup  # noqa: F401


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPOSITORY_ROOT / "conf"


def test_default_config_samples_outcomes_with_natural_mc():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="estimate_ate")

    assert config.estimator._target_ == "naturalv2.estimators.natural_mc.NaturalMC"

    stage_targets = {stage._target_ for stage in config.pipeline.stages.values()}
    assert "naturalv2.pipeline.sample_extraction.SampleTYStage" in stage_targets
    assert (
        "naturalv2.pipeline.conditional_extraction.ConditionalExtractionStage"
        not in stage_targets
    )
