from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import ValidationError

from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME
from naturalv2.pipeline.sample_extraction import (
    _create_sample_ty_response_format,
    _validate_sample_ty_extractions,
)


class ContinuousExperiment:
    options = {TREATMENT_COL_NAME: ["Treatment A"]}

    @staticmethod
    def is_binary_outcome(outcome):
        return False


@pytest.mark.parametrize("outcome", [float("nan"), float("inf"), float("-inf")])
def test_finite_float_response_rejects_non_finite_outcomes(outcome):
    response_format = _create_sample_ty_response_format(
        ContinuousExperiment(), "Outcome A"
    )

    with pytest.raises(ValidationError):
        response_format.model_validate(
            {TREATMENT_COL_NAME: "Treatment A", OUTCOME_COL_NAME: outcome}
        )


def test_finite_float_response_accepts_finite_outcome():
    response_format = _create_sample_ty_response_format(
        ContinuousExperiment(), "Outcome A"
    )

    response = response_format.model_validate(
        {TREATMENT_COL_NAME: "Treatment A", OUTCOME_COL_NAME: 12.5}
    )

    assert getattr(response, OUTCOME_COL_NAME) == 12.5


def test_artifact_validation_rejects_cached_non_finite_outcomes():
    response_format = _create_sample_ty_response_format(
        ContinuousExperiment(), "Outcome A"
    )
    extractions = pd.DataFrame(
        {
            TREATMENT_COL_NAME: ["Treatment A"] * 4,
            OUTCOME_COL_NAME: ["12.5", float("nan"), float("inf"), float("-inf")],
            "report": ["a", "b", "c", "d"],
        },
        index=[10, 11, 12, 13],
    )

    # 3 of 4 rejected is 75%, at or above HIGH_REJECTION_RATE, so this now
    # escalates past warning to error.
    with patch("naturalv2.pipeline.sample_extraction.logger.error") as log_warning:
        validated = _validate_sample_ty_extractions(
            extractions,
            response_format,
            nct_id="NCT012",
            outcome="Outcome A",
        )

    assert validated.index.tolist() == [10]
    assert validated[OUTCOME_COL_NAME].tolist() == [12.5]
    assert validated["report"].tolist() == ["a"]
    # rejected_by_field is now the final positional argument
    assert log_warning.call_args.args[-4:-1] == (4, 3, pytest.approx(0.75))
    assert log_warning.call_args.args[-1] == {OUTCOME_COL_NAME: 3}
    assert log_warning.call_args.kwargs["extra"] == {
        "phase": "sample_ty_artifact_validation",
        "schema_id": "sample_ty_response.v2",
        "status": "rejected",
        "nct_id": "NCT012",
        "outcome": "Outcome A",
        "n_sampled": 4,
        "n_rejected": 3,
        "rejection_rate": pytest.approx(0.75),
        "rejected_by_field": {OUTCOME_COL_NAME: 3},
    }


def test_artifact_validation_raises_when_every_record_is_invalid():
    response_format = _create_sample_ty_response_format(
        ContinuousExperiment(), "Outcome A"
    )
    extractions = pd.DataFrame(
        {
            TREATMENT_COL_NAME: ["Treatment A"],
            OUTCOME_COL_NAME: [float("inf")],
        }
    )

    with pytest.raises(ValueError, match="Every sampled treatment/outcome record"):
        _validate_sample_ty_extractions(
            extractions,
            response_format,
            nct_id="NCT012",
            outcome="Outcome A",
        )


def test_low_rejection_rate_stays_a_warning():
    """Below HIGH_REJECTION_RATE the gate warns rather than errors."""
    response_format = _create_sample_ty_response_format(
        ContinuousExperiment(), "Outcome A"
    )
    # 1 of 50 rejected -> 2%
    extractions = pd.DataFrame(
        {
            TREATMENT_COL_NAME: ["Treatment A"] * 50,
            OUTCOME_COL_NAME: [1.0] * 49 + [float("inf")],
        }
    )
    with (
        patch("naturalv2.pipeline.sample_extraction.logger.error") as log_error,
        patch("naturalv2.pipeline.sample_extraction.logger.warning") as log_warning,
    ):
        _validate_sample_ty_extractions(
            extractions, response_format, nct_id="NCT012", outcome="Outcome A"
        )
    assert log_warning.called
    assert not log_error.called


def test_rejection_reasons_are_counted_per_field():
    """Rejected outcome vs rejected treatment are different problems."""
    response_format = _create_sample_ty_response_format(
        ContinuousExperiment(), "Outcome A"
    )
    extractions = pd.DataFrame(
        {
            TREATMENT_COL_NAME: [
                "Treatment A",
                "Not A Treatment",
                "Treatment A",
                "Treatment A",
            ],
            OUTCOME_COL_NAME: [1.0, 2.0, float("inf"), 4.0],
        }
    )
    with patch("naturalv2.pipeline.sample_extraction.logger.error") as log:
        _validate_sample_ty_extractions(
            extractions, response_format, nct_id="NCT012", outcome="Outcome A"
        )
    by_field = log.call_args.kwargs["extra"]["rejected_by_field"]
    assert by_field == {TREATMENT_COL_NAME: 1, OUTCOME_COL_NAME: 1}


def test_a_record_failing_both_fields_counts_against_both():
    """Taking only the first error would undercount."""
    response_format = _create_sample_ty_response_format(
        ContinuousExperiment(), "Outcome A"
    )
    extractions = pd.DataFrame(
        {TREATMENT_COL_NAME: ["Not A Treatment"], OUTCOME_COL_NAME: [float("nan")]}
    )
    with (
        patch("naturalv2.pipeline.sample_extraction.logger.error") as log,
        pytest.raises(ValueError, match="failed artifact validation"),
    ):
        _validate_sample_ty_extractions(
            extractions, response_format, nct_id="NCT012", outcome="Outcome A"
        )
    by_field = log.call_args.kwargs["extra"]["rejected_by_field"]
    assert by_field == {TREATMENT_COL_NAME: 1, OUTCOME_COL_NAME: 1}
