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

    with patch("naturalv2.pipeline.sample_extraction.logger.warning") as log_warning:
        validated = _validate_sample_ty_extractions(
            extractions,
            response_format,
            nct_id="NCT012",
            outcome="Outcome A",
        )

    assert validated.index.tolist() == [10]
    assert validated[OUTCOME_COL_NAME].tolist() == [12.5]
    assert validated["report"].tolist() == ["a"]
    assert log_warning.call_args.args[-3:] == (4, 3, pytest.approx(0.75))
    assert log_warning.call_args.kwargs["extra"] == {
        "phase": "sample_ty_artifact_validation",
        "schema_id": "sample_ty_response.v1",
        "status": "rejected",
        "nct_id": "NCT012",
        "outcome": "Outcome A",
        "n_sampled": 4,
        "n_rejected": 3,
        "rejection_rate": pytest.approx(0.75),
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
