import importlib.resources
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import DictConfig

import naturalv2.pipeline.ownership as ownership_module
from naturalv2.pipeline.constants import TREATMENT_COL_NAME
from naturalv2.pipeline.ownership import (
    OWNERSHIP_COL,
    TreatmentOwnershipGateStage,
)
from naturalv2.prompts.utils import load_prompt


PROMPTS_DIR = str(importlib.resources.files("naturalv2.prompts.templates"))


def make_stage() -> TreatmentOwnershipGateStage:
    stage = TreatmentOwnershipGateStage(
        DictConfig({"model_id": "fake/model"}), name="treatment_ownership"
    )
    stage._llm = object()
    return stage


def reddit_reports() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "report_type": ["submission", "comment", "comment", "comment"],
            "report_text": [
                "Lithium helped me.",
                "I take lithium every night.",
                "Should I take lithium?",
                "That is interesting.",
            ],
            "report": [
                "Original submission report",
                "Original accepted comment report",
                "Original rejected comment report",
                "Original unclear comment report",
            ],
            TREATMENT_COL_NAME: ["Lithium"] * 4,
        }
    )


@pytest.mark.asyncio
async def test_gate_adds_one_decision_and_filters_comments(monkeypatch):
    captured = {}

    async def fake_extract_covariates(**kwargs):
        captured["input"] = kwargs["input_df"].copy()
        response_format = kwargs["response_format"]
        assert response_format(treatment_ownership="yes").model_dump() == {
            OWNERSHIP_COL: "Yes"
        }
        result = kwargs["input_df"].copy()
        result[OWNERSHIP_COL] = ["Yes", "No", "Unclear"]
        return result

    monkeypatch.setattr(ownership_module, "extract_covariates", fake_extract_covariates)
    result = await make_stage().process(
        reddit_reports(), SimpleNamespace(source_name="reddit")
    )

    assert list(result.index) == [0, 1]
    assert result[OWNERSHIP_COL].tolist() == ["Yes", "Yes"]
    assert result.loc[1, "report"] == "Original accepted comment report"
    gate_input = captured["input"]
    assert list(gate_input.index) == [1, 2, 3]
    assert gate_input.loc[1, "report"] == (
        "SELECTED TREATMENT: Lithium\nTARGET COMMENT:\nI take lithium every night."
    )
    data = pd.DataFrame({"report": ["A publication report"]})
    bypassed = await make_stage().process(data, SimpleNamespace(source_name="pubmed"))
    pd.testing.assert_frame_equal(bypassed, data)


def test_ownership_prompt_is_a_single_narrow_decision():
    prompt = load_prompt(
        PROMPTS_DIR,
        "treatment_ownership",
        report="SELECTED TREATMENT: Lithium\nTARGET COMMENT:\nI take it daily.",
    )

    assert '"treatment_ownership": "Yes | No | Unclear"' in prompt
    assert "Use only the target comment as ownership evidence" in prompt
    assert "future plans" in prompt
