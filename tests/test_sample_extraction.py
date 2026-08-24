import logging
from unittest.mock import Mock, patch

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


def test_response_schema_rejects_non_finite_outcome():
    experiment = Mock(
        options={TREATMENT_COL_NAME: ["Treatment A"]},
        is_binary_outcome=Mock(return_value=False),
    )
    response_format = _create_sample_ty_response_format(experiment, OUTCOME)

    with pytest.raises(ValidationError):
        response_format.model_validate(
            {TREATMENT_COL_NAME: "Treatment A", OUTCOME_COL_NAME: float("inf")}
        )


def test_cached_artifact_validation_filters_all_invalid_outcomes():
    extractions = pd.DataFrame(
        {
            OUTCOME_COL_NAME: [-1, 0, 55, 56, pd.NA, float("inf")],
            "report": list("abcdef"),
        },
        index=[10, 11, 12, 13, 14, 15],
    )

    validated = _validate(
        extractions,
        policy=ALLOW_HIGH_REJECTION_POLICY,
        bounds=BOUNDS,
    )

    assert validated.index.tolist() == [11, 12]
    assert validated[OUTCOME_COL_NAME].tolist() == [0, 55]
    assert validated["report"].tolist() == ["b", "c"]


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


def test_rejection_reasons_separate_every_cause():
    """Four causes, four counts -- they call for different responses."""
    extractions = pd.DataFrame(
        {
            OUTCOME_COL_NAME: [
                10.0,             # kept
                "not-a-number",   # unparsed
                float("nan"),     # unparsed
                float("inf"),     # infinite
                -1.0,             # below minimum
                4_444_000.0,      # above maximum
            ]
        }
    )
    with patch("naturalv2.pipeline.sample_extraction.logger.error") as log:
        with pytest.raises(ValueError, match="high-rejection threshold"):
            _filter_invalid_sampled_outcomes(
                extractions,
                nct_id="NCT012",
                outcome="Outcome A",
                bounds=BOUNDS,
                sample_validation=TEN_PERCENT_POLICY,
            )
    assert log.call_args.kwargs["extra"]["rejection_reasons"] == {
        "unparsed": 2,
        "infinite": 1,
        "below_minimum": 1,
        "above_maximum": 1,
    }


def test_change_endpoint_bounds_keep_improvements():
    """A change endpoint spans the instrument's width signed, not its level range."""
    # -10 is a real improvement on a 0-55 instrument scored as change.
    extractions = pd.DataFrame({OUTCOME_COL_NAME: [-10.0, 5.0, 20.0]})
    kept = _filter_invalid_sampled_outcomes(
        extractions,
        nct_id="NCT06366724",
        outcome="Functional Capacity",
        bounds=OutcomeBounds(minimum=-55, maximum=55),
        sample_validation=TEN_PERCENT_POLICY,
    )
    assert kept[OUTCOME_COL_NAME].tolist() == [-10.0, 5.0, 20.0]
