"""Tests for inclusion probability stages (logit-scored vs sample-based)."""

import importlib.resources
import math
import os
import re
from ast import literal_eval

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv
from omegaconf import DictConfig

from naturalv2.cli.estimate_ate import _weight_by_inclusion
from naturalv2.models.lm import Model
from naturalv2.models.types import LogProbs, ModelResponse, TokenUsage
from naturalv2.models.utils import TokenTracker
from naturalv2.pipeline.conditional_extraction import (
    ConditionalsExtractType,
    extract_conditionals,
)
from naturalv2.pipeline.constants import INCLUSION_COL_NAME
from naturalv2.pipeline.natural import PipelineContext
from naturalv2.pipeline.sample_extraction import SampledInclusionProbStage
from naturalv2.prompts.utils import load_prompt


load_dotenv()

OUTCOME = "sleep_quality"
SOURCE = "reddit"
PROMPTS_DIR = str(importlib.resources.files("naturalv2.prompts.templates"))
SAMPLED_COL = f"{INCLUSION_COL_NAME}_sampled"


class FakeExperiment:
    """Minimal Experiment stand-in that renders the real prompt templates."""

    nct_id = "NCT00000000"
    inclusion_criteria = (
        "Inclusion Criteria: Adults aged 18 to 65 years with self-reported insomnia. "
        "Exclusion Criteria: Children under 18 years of age; pregnancy."
    )
    options = {INCLUSION_COL_NAME: ["No", "Yes"]}

    def get_question_prompts(self):
        return {
            INCLUSION_COL_NAME: load_prompt(
                PROMPTS_DIR,
                "question_inclusion",
                return_format="prompt",
                inclusion_criteria=self.inclusion_criteria,
            )
        }

    def build_prompt_for_report(
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
            inclusion_criteria=self.inclusion_criteria,
        )


class FakeModelBase(Model):
    """Records calls; subclasses script the async responses."""

    def __init__(self):
        super().__init__("fake/fake-model")
        self.calls = []

    def invoke(self, input_data, *args, **kwargs):
        raise NotImplementedError


class FakeLogitModel(FakeModelBase):
    """Scores 'A: Yes' prompt variants ln(3) higher -> softmax [0.25, 0.75]."""

    async def ainvoke(self, input_data, *args, **kwargs):
        self.calls.append(input_data)
        score = math.log(3) if input_data.endswith("A: Yes") else 0.0
        return ModelResponse(
            model_id=self.model_id,
            output_text="",
            prompt_logprobs=LogProbs(tokens=["x"], logprobs=[score]),
        )


class FakeVoteModel(FakeModelBase):
    """Scripted Yes/No answers per report ("Report <i>"); None means unparseable."""

    def __init__(self, answers_by_report):
        super().__init__()
        if isinstance(answers_by_report, list):
            answers_by_report = {0: answers_by_report}
        self.answers = {key: list(votes) for key, votes in answers_by_report.items()}

    async def ainvoke(self, input_data, *args, response_format=None, **kwargs):
        self.calls.append(input_data)
        report_idx = int(re.search(r"Report (\d+)", input_data[-1]["content"]).group(1))
        answer = self.answers[report_idx].pop(0)
        parsed = (
            response_format(**{INCLUSION_COL_NAME: answer})
            if answer is not None
            else None
        )
        return ModelResponse(
            model_id=self.model_id,
            output_text=str(answer),
            output_parsed=parsed,
            token_usage=TokenUsage(10, 1, 11),
        )


def make_stage(fake_llm, **overrides):
    stage = SampledInclusionProbStage(
        DictConfig({"model_id": "fake/fake-model"}), name="inclusion_prob", **overrides
    )
    stage._llm = fake_llm
    return stage


def make_context(tmp_path):
    return PipelineContext(
        experiment=FakeExperiment(),
        source_name=SOURCE,
        estimator_type="NaturalIPW",
        outcome=OUTCOME,
        save_path=str(tmp_path),
        exp_name="testexp",
        _token_tracker=TokenTracker(),
    )


def make_reports(n):
    return pd.DataFrame(
        {
            "report": [
                f"Report {i}: I am 30 years old and have insomnia." for i in range(n)
            ]
        }
    )


def make_extracted(votes_by_index):
    """Extraction frame as extract_covariates would return it for given votes."""
    rows = [
        {"source_index": idx, INCLUSION_COL_NAME: vote}
        for idx, votes in votes_by_index.items()
        for vote in votes
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Vote aggregation
# ---------------------------------------------------------------------------


def test_aggregate_vote_fraction():
    stage = make_stage(None, seed=0)
    out = stage._aggregate(
        make_extracted({0: ["Yes"] * 6 + ["No"] * 2}), make_reports(1), ["No", "Yes"]
    )
    assert literal_eval(out["inclusion_probs"].iloc[0]) == pytest.approx([0.25, 0.75])
    assert out["num_valid_votes"].iloc[0] == 8


def test_aggregate_case_insensitive():
    stage = make_stage(None, seed=0)
    out = stage._aggregate(
        make_extracted({0: ["yes", "NO", "Yes", "no"]}), make_reports(1), ["No", "Yes"]
    )
    assert literal_eval(out["inclusion_probs"].iloc[0]) == pytest.approx([0.5, 0.5])


def test_aggregate_counts_only_surviving_votes():
    # extract_covariates drops unparseable votes before aggregation; the
    # fraction is over whatever survived.
    stage = make_stage(None, num_samples=8, seed=0)
    out = stage._aggregate(
        make_extracted({0: ["Yes"] * 3 + ["No"] * 3}), make_reports(1), ["No", "Yes"]
    )
    assert literal_eval(out["inclusion_probs"].iloc[0]) == pytest.approx([0.5, 0.5])
    assert out["num_valid_votes"].iloc[0] == 6


def test_aggregate_drops_report_with_no_votes():
    stage = make_stage(None, seed=0)
    out = stage._aggregate(
        make_extracted({0: ["Yes"] * 2}), make_reports(2), ["No", "Yes"]
    )
    assert list(out.index) == [0]


# ---------------------------------------------------------------------------
# Sampled stage end-to-end (mocked LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vote_stage_end_to_end(tmp_path):
    fake = FakeVoteModel({0: ["Yes"] * 6 + ["No"] * 2, 1: ["Yes"] * 8, 2: ["No"] * 8})
    stage = make_stage(fake, num_samples=8, seed=0)
    context = make_context(tmp_path)
    out = await stage.process(make_reports(3), context)

    assert len(out) == 3
    probs = [literal_eval(p) for p in out["inclusion_probs"]]
    assert probs[0] == pytest.approx([0.25, 0.75])
    assert probs[1] == pytest.approx([0.0, 1.0])
    assert probs[2] == pytest.approx([1.0, 0.0])
    assert set(out[SAMPLED_COL]) <= {"No", "Yes"}
    assert (out["num_valid_votes"] == 8).all()
    assert "report" in out.columns  # original columns preserved

    # Per-vote extractions saved under a filename distinct from the logit stage
    expected = tmp_path / "results" / "NCT00000000_testexp"
    assert (expected / f"fake-model_sample_inclusion_{OUTCOME}.csv").exists()

    # Token usage tracked: 3 reports x 8 votes x 11 tokens
    assert (
        context._token_tracker.get_stage_stats("inclusion_prob")["total_tokens"] == 264
    )

    # Weights flow into _weight_by_inclusion unchanged
    weighted = _weight_by_inclusion(np.array([[1.0, 2.0, 3.0]]), out)
    assert weighted[0] == pytest.approx(
        np.average([1.0, 2.0, 3.0], weights=[0.75, 1.0, 0.0])
    )


@pytest.mark.asyncio
async def test_unparseable_votes_dropped(tmp_path):
    fake = FakeVoteModel({0: ["Yes", "Yes", "No", None], 1: [None] * 4})
    stage = make_stage(fake, num_samples=4, seed=0)
    out = await stage.process(make_reports(2), make_context(tmp_path))

    assert list(out.index) == [0]  # report 1 had no valid votes
    assert literal_eval(out["inclusion_probs"].iloc[0])[1] == pytest.approx(2 / 3)
    assert out["num_valid_votes"].iloc[0] == 3


@pytest.mark.asyncio
async def test_resume_skips_existing_votes(tmp_path):
    first = FakeVoteModel({0: ["Yes"] * 4, 1: ["Yes"] * 4})
    await make_stage(first, num_samples=4, seed=0).process(
        make_reports(2), make_context(tmp_path)
    )
    assert len(first.calls) == 8

    # A rerun over a superset only issues the new report's votes; the fake
    # would KeyError if reports 0 or 1 were re-processed.
    second = FakeVoteModel({2: ["No"] * 4})
    out = await make_stage(second, num_samples=4, seed=0).process(
        make_reports(3), make_context(tmp_path)
    )
    assert len(second.calls) == 4
    assert len(out) == 3


@pytest.mark.asyncio
async def test_sampled_label_deterministic_with_seed(tmp_path):
    outs = []
    for sub in ["a", "b"]:
        fake = FakeVoteModel({i: ["Yes"] * 4 + ["No"] * 4 for i in range(3)})
        out = await make_stage(fake, num_samples=8, seed=7).process(
            make_reports(3), make_context(tmp_path / sub)
        )
        outs.append(out[SAMPLED_COL].tolist())
    assert outs[0] == outs[1]


# ---------------------------------------------------------------------------
# Logit vs sampled comparison
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logit_and_sampled_paths_agree(tmp_path):
    """Same input, equivalent evidence -> same inclusion_probs contract and weights."""
    experiment = FakeExperiment()
    df = make_reports(1)

    logit_llm = FakeLogitModel()
    logit_df, _ = await extract_conditionals(
        df,
        experiment,
        SOURCE,
        OUTCOME,
        logit_llm,
        "fake-logit",
        str(tmp_path),
        "testexp",
        ConditionalsExtractType.INCLUSION,
        is_offline_inference=False,
    )

    vote_llm = FakeVoteModel({0: ["Yes"] * 6 + ["No"] * 2})
    sampled_df = await make_stage(vote_llm, num_samples=8, seed=0).process(
        df, make_context(tmp_path)
    )

    # Both prompts contain the report, the inclusion criteria, the same
    # question sentence and options; the logit variants differ only in the
    # teacher-forced answer.
    question_sentence = (
        "Does the patient described in the report satisfy these criteria?"
    )
    report = df["report"].iloc[0]
    assert logit_llm.calls[0].endswith("A: No")
    assert logit_llm.calls[1].endswith("A: Yes")
    assert logit_llm.calls[0].removesuffix("No") == logit_llm.calls[1].removesuffix(
        "Yes"
    )
    sampled_user = vote_llm.calls[0][-1]["content"]
    for prompt in [*logit_llm.calls, sampled_user]:
        assert report in prompt
        assert experiment.inclusion_criteria in prompt
        assert question_sentence in prompt
        assert "Options: a) No b) Yes" in prompt

    # Same output contract and identical downstream weights
    logit_probs = literal_eval(logit_df["inclusion_probs"].iloc[0])
    sampled_probs = literal_eval(sampled_df["inclusion_probs"].iloc[0])
    assert logit_probs == pytest.approx([0.25, 0.75])
    assert sampled_probs == pytest.approx([0.25, 0.75])
    assert SAMPLED_COL in logit_df.columns and SAMPLED_COL in sampled_df.columns
    weights = np.array([[1.0]])
    assert _weight_by_inclusion(weights, logit_df) == pytest.approx(
        _weight_by_inclusion(weights, sampled_df)
    )


# ---------------------------------------------------------------------------
# Real-API integration test (skipped without API key)
# ---------------------------------------------------------------------------


@pytest.mark.integration_test
@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY/GOOGLE_API_KEY not set",
)
async def test_sampled_inclusion_real_api(tmp_path):
    """The matching report should get a higher P(inclusion) than the non-matching one."""
    reports = pd.DataFrame(
        {
            "report": [
                "I'm a 32 year old office worker and I've had insomnia for three "
                "years. I barely sleep four hours a night and tried melatonin "
                "without success.",
                "My 8 year old daughter has trouble falling asleep before school "
                "days. The pediatrician says it's common at her age.",
            ]
        }
    )
    stage = SampledInclusionProbStage(
        DictConfig(
            {
                "_target_": "naturalv2.models.lm.LiteLLMModel",
                "model_id": "gemini/gemini-flash-lite-latest",
            }
        ),
        name="inclusion_prob",
        num_samples=4,
        call_kwargs={"temperature": 1.0},
        seed=0,
    )
    out = await stage.process(reports, make_context(tmp_path))
    p_yes = [literal_eval(p)[1] for p in out["inclusion_probs"]]
    assert p_yes[0] > p_yes[1]
