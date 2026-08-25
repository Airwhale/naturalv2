from datetime import datetime, timezone
from importlib import resources
from unittest.mock import patch

import polars as pl

from naturalv2.prompts.utils import load_prompt
from naturalv2.sources.components.helpers import build_treatment_automaton
from naturalv2.sources.reddit.stages.curate import _process_chunk, _SubredditContext


def _context() -> _SubredditContext:
    return _SubredditContext(
        name="testsub",
        automaton=build_treatment_automaton(["lithium"]),
        term_to_experiments={"lithium": {"NCT001"}},
        experiment_publication_dates={
            "NCT001": datetime.max.replace(tzinfo=timezone.utc)
        },
        global_max_date=datetime.max.replace(tzinfo=timezone.utc),
        experiment_to_terms={"NCT001": ["lithium"]},
    )


def test_process_chunk_matches_treatments_only_in_subject_text():
    data = pl.DataFrame(
        {
            "subreddit": ["testsub"] * 3,
            "title": ["Lithium experience"] * 3,
            "initial_post": ["Lithium cleared my brain fog."] * 3,
            "report_text": [
                "Have you tried an antihistamine?",
                "Lithium helped my fatigue.",
                "It helped my fatigue.",
            ],
            "report_type": ["comment", "comment", "submission"],
            "date_created": ["2026-01-01"] * 3,
            "permalink": ["context-only", "own-comment", "own-submission"],
        }
    )

    result = _process_chunk(_context(), data, filter_by_date=False)

    assert result is not None
    assert set(result["permalink"]) == {"own-comment", "own-submission"}


def test_comment_report_presents_target_patient_before_thread_context():
    data = pl.DataFrame(
        {
            "subreddit": ["testsub"],
            "title": ["Lithium experience"],
            "initial_post": ["Lithium cleared my brain fog."],
            "report_text": ["Lithium helped my fatigue."],
            "report_type": ["comment"],
            "date_created": ["2026-01-01"],
            "permalink": ["own-comment"],
        }
    )

    result = _process_chunk(_context(), data, filter_by_date=False)

    assert result is not None
    report = result.item(0, "report")
    assert (
        "Attribute treatment, covariates, and outcomes only to this commenter" in report
    )
    assert report.index("**Comment from the target patient**") < report.index(
        "**Thread context (not target-patient evidence)**"
    )
    assert (
        "The initial post is context, not evidence about the target patient" in report
    )


def test_sample_ty_prompt_binds_answers_to_target_patient():
    prompts_dir = str(resources.files("naturalv2.prompts.templates"))
    messages = load_prompt(
        prompts_dir,
        "sample_ty",
        return_format="messages",
        outcome="Brain Fog",
        outcome_desc="Change in brain fog",
        source="Reddit",
        report="Example report",
        covariates=["age"],
        covariate_answers={"age": "40"},
        treatment_options=["Lithium", "Placebo"],
        outcome_is_binary=True,
        outcome_options=["No", "Yes"],
        outcome_timeframe=None,
    )

    assert isinstance(messages, list)
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert (
        'author of the section labeled "Comment from the target patient"'
        in system_prompt
    )
    assert "future plan" in system_prompt
    assert "Do not invent an event" in system_prompt
    assert '"treatment_taken"' in user_prompt
    assert '"outcome_category"' in user_prompt
    assert "Do not invent additional facts" in user_prompt
    assert "report subject" not in system_prompt + user_prompt


def test_thin_evidence_is_flagged():
    """Binding evidence to the commenter can leave a small trial unusable."""
    from naturalv2.sources.reddit.stages.curate import (
        MIN_USABLE_EVIDENCE,
        _warn_on_thin_evidence,
    )

    with patch("naturalv2.sources.reddit.stages.curate.logger.warning") as warn:
        _warn_on_thin_evidence({"NCT_THIN": 1, "NCT_OK": MIN_USABLE_EVIDENCE})

    assert warn.call_count == 1
    assert warn.call_args.kwargs["extra"]["nct_id"] == "NCT_THIN"
    assert warn.call_args.kwargs["extra"]["n_curated"] == 1


def test_sufficient_evidence_is_silent():
    from naturalv2.sources.reddit.stages.curate import (
        MIN_USABLE_EVIDENCE,
        _warn_on_thin_evidence,
    )

    with patch("naturalv2.sources.reddit.stages.curate.logger.warning") as warn:
        _warn_on_thin_evidence({"NCT_OK": MIN_USABLE_EVIDENCE + 5})
    assert not warn.called
