import pytest
from pydantic import ValidationError

from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME
from naturalv2.pipeline.sample_extraction import _create_sample_ty_response_format


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
