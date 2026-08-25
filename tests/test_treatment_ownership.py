import importlib.resources

import pandas as pd
import pytest
from omegaconf import DictConfig

import naturalv2.pipeline.ownership as ownership_module
from naturalv2.models.lm import Model
from naturalv2.models.types import ModelResponse, TokenUsage
from naturalv2.models.utils import TokenTracker
from naturalv2.pipeline.constants import OUTCOME_COL_NAME, TREATMENT_COL_NAME
from naturalv2.pipeline.natural import PipelineContext
from naturalv2.pipeline.ownership import (
    FIRSTHAND,
    NOT_APPLICABLE,
    OWNERSHIP_DECISION_COL,
    OWNERSHIP_EVIDENCE_COL,
    OWNERSHIP_REASON_COL,
    OWNERSHIP_TREATMENT_COL,
    OwnershipAwareSampleTYStage,
    TreatmentOwnershipGateStage,
    _mentions_target_treatment,
)
from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.components.helpers import build_treatment_automaton


OUTCOME = "Brain Fog"
PROMPTS_DIR = str(importlib.resources.files("naturalv2.prompts.templates"))


class FakeExperiment:
    nct_id = "NCT00000000"
    covariate_names: list[str] = []
    treatment_names = ["Lithium", "Placebo"]
    options = {TREATMENT_COL_NAME: treatment_names}
    treatment_common_names = {"reddit": {"Lithium": ["lithium orotate"], "Placebo": []}}
    outcome_desc = {OUTCOME: "Whether the patient's brain fog improved."}

    def get_all_treatment_names_for_source(self, source):
        return ["Lithium", "lithium orotate", "Placebo"]

    def is_binary_outcome(self, outcome):
        return True

    def apply_transform(self, values, repr_type="language"):
        return values

    def discretize_ty(self, data, outcome):
        result = data.copy()
        treatment_map = {name: index for index, name in enumerate(self.treatment_names)}
        result[f"{TREATMENT_COL_NAME}_discretized"] = result[TREATMENT_COL_NAME].map(
            treatment_map
        )
        result[f"{OUTCOME_COL_NAME}_discretized"] = result[OUTCOME_COL_NAME].map(
            {"No": 0, "Yes": 1}
        )
        return result

    def build_prompt_for_report(  # noqa: PLR0917
        self,
        prompt_type,
        outcome,
        source_name,
        report,
        covariate_answers=None,
        return_format="messages",
    ):
        return load_prompt(
            PROMPTS_DIR,
            prompt_type,
            return_format=return_format,
            source=source_name,
            report=report,
            outcome=outcome,
            outcome_desc=self.outcome_desc[outcome],
            outcome_is_binary=True,
            outcome_options=["No", "Yes"],
            outcome_timeframe=None,
            covariates=self.covariate_names,
            covariate_answers=covariate_answers or {},
            treatment_options=self.treatment_names,
            treatments=self.treatment_common_names.get(source_name, {}),
        )


class FakeModelBase(Model):
    def __init__(self):
        super().__init__("fake/fake-model")
        self.calls = []

    def invoke(self, input_data, *args, **kwargs):
        raise NotImplementedError


class FakeOwnershipModel(FakeModelBase):
    async def ainvoke(self, input_data, *args, response_format=None, **kwargs):
        self.calls.append(input_data)
        user_prompt = input_data[-1]["content"]
        target_comment = user_prompt.split(
            "TARGET COMMENT (written by the target patient):", maxsplit=1
        )[1].split("THREAD CONTEXT", maxsplit=1)[0]

        if "Should I take lithium" in target_comment:
            values = {
                OWNERSHIP_DECISION_COL: "Not firsthand",
                OWNERSHIP_TREATMENT_COL: "Unknown",
                OWNERSHIP_REASON_COL: "question",
                OWNERSHIP_EVIDENCE_COL: "Should I take lithium?",
            }
        elif "plan to take lithium" in target_comment:
            values = {
                OWNERSHIP_DECISION_COL: "Not firsthand",
                OWNERSHIP_TREATMENT_COL: "Unknown",
                OWNERSHIP_REASON_COL: "future_plan",
                OWNERSHIP_EVIDENCE_COL: "plan to take lithium next week",
            }
        elif "I take lithium" in target_comment:
            values = {
                OWNERSHIP_DECISION_COL: FIRSTHAND,
                OWNERSHIP_TREATMENT_COL: "Lithium",
                OWNERSHIP_REASON_COL: "current_use",
                OWNERSHIP_EVIDENCE_COL: "I take lithium every night",
            }
        else:
            values = {
                OWNERSHIP_DECISION_COL: "Not firsthand",
                OWNERSHIP_TREATMENT_COL: "Unknown",
                OWNERSHIP_REASON_COL: "third_person",
                OWNERSHIP_EVIDENCE_COL: "My sister takes lithium",
            }

        return ModelResponse(
            model_id=self.model_id,
            output_text=str(values),
            output_parsed=response_format(**values),
            token_usage=TokenUsage(20, 5, 25),
        )


class FakeSampleTYModel(FakeModelBase):
    async def ainvoke(self, input_data, *args, response_format=None, **kwargs):
        self.calls.append(input_data)
        values = {TREATMENT_COL_NAME: "Placebo", OUTCOME_COL_NAME: "Yes"}
        return ModelResponse(
            model_id=self.model_id,
            output_text=str(values),
            output_parsed=response_format(**values),
            token_usage=TokenUsage(20, 5, 25),
        )


def make_context(tmp_path, source="reddit"):
    return PipelineContext(
        experiment=FakeExperiment(),
        source_name=source,
        estimator_type="NaturalMC",
        outcome=OUTCOME,
        save_path=str(tmp_path),
        exp_name="ownership-test",
        _token_tracker=TokenTracker(),
    )


def make_gate(fake_llm):
    stage = TreatmentOwnershipGateStage(
        DictConfig({"model_id": "fake/fake-model"}), name="treatment_ownership"
    )
    stage._llm = fake_llm
    return stage


def make_reddit_reports():
    rows = [
        {
            "report_type": "submission",
            "report_text": "Lithium helped my fatigue.",
            "initial_post": "",
            "title": "My experience",
            "report": "Original submission report",
        },
        {
            "report_type": "comment",
            "report_text": "Have you tried an antihistamine?",
            "initial_post": "Lithium helped my fatigue.",
            "title": "Lithium experience",
            "report": "Original context-only comment report",
        },
        {
            "report_type": "comment",
            "report_text": "I take lithium every night.",
            "initial_post": "What helped your fatigue?",
            "title": "Fatigue treatments",
            "report": "Original firsthand comment report",
        },
        {
            "report_type": "comment",
            "report_text": "Should I take lithium?",
            "initial_post": "Lithium helped my fatigue.",
            "title": "Lithium experience",
            "report": "Original question comment report",
        },
        {
            "report_type": "comment",
            "report_text": "I plan to take lithium next week.",
            "initial_post": "Lithium helped my fatigue.",
            "title": "Lithium experience",
            "report": "Original future-plan comment report",
        },
        {
            "report_type": "comment",
            "report_text": "My sister takes lithium.",
            "initial_post": "Lithium helped my fatigue.",
            "title": "Lithium experience",
            "report": "Original third-person comment report",
        },
    ]
    return pd.DataFrame(rows)


def test_lexical_gate_normalizes_aliases_and_ldn_boundaries():
    automaton = build_treatment_automaton(["Lithium", "LDN"])

    assert _mentions_target_treatment("LITHIUM helped me", automaton)
    assert _mentions_target_treatment("I take LDN", automaton)
    assert not _mentions_target_treatment("golden sunlight", automaton)


@pytest.mark.asyncio
async def test_ownership_gate_filters_and_subject_binds_comments(tmp_path):
    fake = FakeOwnershipModel()
    stage = make_gate(fake)

    result = await stage.process(make_reddit_reports(), make_context(tmp_path))

    assert list(result.index) == [0, 2]
    assert len(fake.calls) == 4
    assert result.loc[0, OWNERSHIP_DECISION_COL] == NOT_APPLICABLE
    assert result.loc[0, "report"] == "Original submission report"
    assert result.loc[2, OWNERSHIP_DECISION_COL] == FIRSTHAND
    assert result.loc[2, OWNERSHIP_TREATMENT_COL] == "Lithium"
    assert result.loc[2, "report"] == "I take lithium every night."
    assert "THREAD CONTEXT" in fake.calls[0][-1]["content"]
    assert stage.get_stats()["comments_lexically_rejected"] == 1
    assert stage.get_stats()["comments_accepted"] == 1


@pytest.mark.asyncio
async def test_non_reddit_source_bypasses_gate(tmp_path):
    fake = FakeOwnershipModel()
    stage = make_gate(fake)
    data = pd.DataFrame({"report": ["A publication report"]})

    result = await stage.process(data, make_context(tmp_path, source="pubmed"))

    pd.testing.assert_frame_equal(result, data)
    assert not fake.calls


@pytest.mark.asyncio
async def test_reddit_gate_requires_contextualized_columns(tmp_path):
    stage = make_gate(FakeOwnershipModel())

    with pytest.raises(ValueError, match="report_text, report_type"):
        await stage.process(
            pd.DataFrame({"report": ["Incomplete"]}), make_context(tmp_path)
        )


@pytest.mark.asyncio
async def test_reddit_gate_rejects_stale_ownership_artifact(tmp_path, monkeypatch):
    async def return_incomplete_artifact(**kwargs):
        return pd.DataFrame({"report": ["I take lithium."]})

    monkeypatch.setattr(
        ownership_module, "extract_covariates", return_incomplete_artifact
    )
    stage = make_gate(FakeOwnershipModel())
    data = make_reddit_reports().loc[[2]]

    with pytest.raises(RuntimeError, match="fresh experiment name"):
        await stage.process(data, make_context(tmp_path))


@pytest.mark.asyncio
async def test_sample_ty_uses_gate_treatment_as_final_answer(tmp_path):
    fake = FakeSampleTYModel()
    stage = OwnershipAwareSampleTYStage(
        DictConfig({"model_id": "fake/fake-model"}), name="sample_ty"
    )
    stage._llm = fake
    data = pd.DataFrame(
        {
            "report": ["I take lithium.", "A placebo submission."],
            OWNERSHIP_DECISION_COL: [FIRSTHAND, NOT_APPLICABLE],
            OWNERSHIP_TREATMENT_COL: ["Lithium", "Unknown"],
            OWNERSHIP_REASON_COL: ["current_use", "not_applicable"],
            OWNERSHIP_EVIDENCE_COL: ["I take lithium", ""],
        }
    )

    result = await stage.process(data, make_context(tmp_path))

    assert list(result[TREATMENT_COL_NAME]) == ["Lithium", "Placebo"]
    assert list(result[f"{TREATMENT_COL_NAME}_discretized"]) == [0, 1]
    assert list(result[OUTCOME_COL_NAME]) == ["Yes", "Yes"]


def test_ownership_prompt_keeps_comment_and_context_roles_separate():
    messages = FakeExperiment().build_prompt_for_report(
        prompt_type="treatment_ownership",
        outcome=OUTCOME,
        source_name="reddit",
        report=(
            "TARGET COMMENT (written by the target patient):\nI take lithium.\n\n"
            "THREAD CONTEXT (written by another person):\nSomeone else's report."
        ),
    )

    combined = "\n".join(message["content"] for message in messages)
    assert "THREAD CONTEXT" in combined
    assert "never evidence that the target patient used a treatment" in combined
    assert "future plan" in combined
    assert '"Unclear"' in combined
