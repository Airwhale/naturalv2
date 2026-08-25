import importlib.resources
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import DictConfig

import naturalv2.pipeline.ownership as ownership_module
from naturalv2.pipeline.constants import TREATMENT_COL_NAME
from naturalv2.pipeline.ownership import OWNERSHIP_COL, TreatmentOwnershipGateStage
from naturalv2.prompts.utils import load_prompt


@pytest.mark.asyncio
async def test_ownership_check_updates_one_field_and_gates(monkeypatch):
    async def fake_extract(input_df, response_format, **kwargs):
        assert list(input_df.index) == [1, 2]
        assert input_df.loc[1, "report"].startswith(
            "SELECTED TREATMENT: Lithium\nTARGET COMMENT:\n"
        )
        assert response_format(author_used_selected_treatment="yes").model_dump() == {
            OWNERSHIP_COL: "Yes"
        }
        result = input_df.copy()
        result[OWNERSHIP_COL] = ["Yes", "No"]
        return result

    monkeypatch.setattr(ownership_module, "extract_covariates", fake_extract)
    data = pd.DataFrame(
        {
            "report_type": ["submission", "comment", "comment"],
            "report_text": ["Lithium helped.", "I take lithium.", "Should I?"],
            "report": ["submission report", "accepted report", "rejected report"],
            TREATMENT_COL_NAME: ["Lithium"] * 3,
        }
    )
    stage = TreatmentOwnershipGateStage(
        DictConfig({"model_id": "fake/model"}), name="treatment_ownership"
    )
    stage._llm = object()

    result = await stage.process(data, SimpleNamespace(source_name="reddit"))

    assert list(result.index) == [0, 1]
    assert result[OWNERSHIP_COL].tolist() == ["Yes", "Yes"]
    assert result["report"].tolist() == ["submission report", "accepted report"]


def test_ownership_prompt_contract():
    prompt = load_prompt(
        str(importlib.resources.files("naturalv2.prompts.templates")),
        "treatment_ownership",
        report="SELECTED TREATMENT: Lithium\nTARGET COMMENT:\nI take it.",
    )

    assert '"author_used_selected_treatment": "Yes | No | Unclear"' in prompt
    assert "Use only the target comment as ownership evidence" in prompt
