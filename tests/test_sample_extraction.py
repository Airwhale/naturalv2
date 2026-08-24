import pandas as pd
import pytest
from pydantic import ValidationError

from naturalv2.pipeline import (
    OUTCOME_COL_NAME,
    TREATMENT_COL_NAME,
    SampleValidationConfig,
)
from naturalv2.pipeline.sample_extraction import (
    _create_sample_ty_response_format,
    _filter_nonfinite_sampled_outcomes,
)


NCT_ID = "NCT012"
OUTCOME = "Outcome A"
TEN_PERCENT_POLICY = SampleValidationConfig(high_rejection_rate=0.10)
ALLOW_HIGH_REJECTION_POLICY = SampleValidationConfig(
    high_rejection_rate=0.10,
    allow_high_rejection_rate=True,
)


class ContinuousExperiment:
    options = {TREATMENT_COL_NAME: ["Treatment A"]}

    @staticmethod
    def is_binary_outcome(_outcome: str) -> bool:
        return False


RESPONSE_FORMAT = _create_sample_ty_response_format(ContinuousExperiment(), OUTCOME)


def _validate(
    extractions: pd.DataFrame,
    policy: SampleValidationConfig = TEN_PERCENT_POLICY,
) -> pd.DataFrame:
    return _filter_nonfinite_sampled_outcomes(
        extractions,
        nct_id=NCT_ID,
        outcome=OUTCOME,
        sample_validation=policy,
    )


@pytest.mark.parametrize("outcome", [float("nan"), float("inf"), float("-inf")])
def test_response_schema_rejects_non_finite_outcomes(outcome: float):
    with pytest.raises(ValidationError):
        RESPONSE_FORMAT.model_validate(
            {TREATMENT_COL_NAME: "Treatment A", OUTCOME_COL_NAME: outcome}
        )


def test_artifact_validation_removes_cached_non_finite_outcomes():
    extractions = pd.DataFrame(
        {
            TREATMENT_COL_NAME: ["Treatment A"] * 4,
            OUTCOME_COL_NAME: ["12.5", float("nan"), float("inf"), float("-inf")],
            "report": ["a", "b", "c", "d"],
        },
        index=[10, 11, 12, 13],
    )

    validated = _validate(extractions, ALLOW_HIGH_REJECTION_POLICY)

    assert validated.index.tolist() == [10]
    assert validated[OUTCOME_COL_NAME].tolist() == [12.5]
    assert validated["report"].tolist() == ["a"]


def test_high_rejection_rate_blocks_estimation():
    extractions = pd.DataFrame(
        {
            TREATMENT_COL_NAME: ["Treatment A"] * 10,
            OUTCOME_COL_NAME: [1.0] * 9 + [float("inf")],
        }
    )

    with pytest.raises(ValueError, match="high-rejection threshold"):
        _validate(extractions)


def test_all_invalid_records_fail_even_with_override():
    extractions = pd.DataFrame(
        {
            TREATMENT_COL_NAME: ["Treatment A"],
            OUTCOME_COL_NAME: [float("inf")],
        }
    )

    with pytest.raises(ValueError, match="No finite sampled outcomes remain"):
        _validate(extractions, ALLOW_HIGH_REJECTION_POLICY)
