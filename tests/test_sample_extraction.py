from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import ValidationError

from naturalv2.outcome_metadata import OutcomeBounds
from naturalv2.pipeline import (
    OUTCOME_COL_NAME,
    TREATMENT_COL_NAME,
    SampleValidationConfig,
)
from naturalv2.pipeline.sample_extraction import (
    _create_sample_ty_response_format,
    _filter_invalid_sampled_outcomes,
)


NCT_ID = "NCT012"
OUTCOME = "Functional Capacity"
TEN_PERCENT_POLICY = SampleValidationConfig(high_rejection_rate=0.10)
ALLOW_HIGH_REJECTION_POLICY = SampleValidationConfig(
    high_rejection_rate=0.10,
    allow_high_rejection_rate=True,
)
BOUNDS = OutcomeBounds(minimum=0, maximum=55)


class ContinuousExperiment:
    options = {TREATMENT_COL_NAME: ["Treatment A"]}

    @staticmethod
    def is_binary_outcome(_outcome: str) -> bool:
        return False


RESPONSE_FORMAT = _create_sample_ty_response_format(ContinuousExperiment(), OUTCOME)


def _validate(
    extractions: pd.DataFrame,
    *,
    policy: SampleValidationConfig = TEN_PERCENT_POLICY,
    bounds: OutcomeBounds | None = None,
) -> pd.DataFrame:
    return _filter_invalid_sampled_outcomes(
        extractions,
        nct_id=NCT_ID,
        outcome=OUTCOME,
        bounds=bounds,
        sample_validation=policy,
    )


@pytest.mark.parametrize("outcome", [float("nan"), float("inf"), float("-inf")])
def test_response_schema_rejects_non_finite_outcomes(outcome: float):
    with pytest.raises(ValidationError):
        RESPONSE_FORMAT.model_validate(
            {TREATMENT_COL_NAME: "Treatment A", OUTCOME_COL_NAME: outcome}
        )


def test_cached_artifact_validation_filters_all_invalid_outcomes():
    extractions = pd.DataFrame(
        {
            OUTCOME_COL_NAME: [-1, 0, 55, 56, pd.NA, float("inf")],
            "report": list("abcdef"),
        },
        index=[10, 11, 12, 13, 14, 15],
    )

    with patch("naturalv2.pipeline.sample_extraction.logger.error") as log_error:
        validated = _validate(
            extractions,
            policy=ALLOW_HIGH_REJECTION_POLICY,
            bounds=BOUNDS,
        )

    assert validated.index.tolist() == [11, 12]
    assert validated[OUTCOME_COL_NAME].tolist() == [0, 55]
    assert validated["report"].tolist() == ["b", "c"]
    assert log_error.call_args.kwargs["extra"]["rejection_reasons"] == {
        "nonfinite": 2,
        "below_minimum": 1,
        "above_maximum": 1,
    }


def test_combined_rejection_rate_blocks_estimation():
    extractions = pd.DataFrame(
        {OUTCOME_COL_NAME: ([10.0] * 88 + [float("nan")] * 6 + [56.0] * 6)}
    )

    with pytest.raises(ValueError, match="high-rejection threshold"):
        _validate(extractions, bounds=BOUNDS)


def test_all_invalid_records_fail_even_with_override():
    extractions = pd.DataFrame({OUTCOME_COL_NAME: [float("inf")]})

    with pytest.raises(ValueError, match="No valid sampled outcomes remain"):
        _validate(extractions, policy=ALLOW_HIGH_REJECTION_POLICY, bounds=BOUNDS)
