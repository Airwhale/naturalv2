import pytest
from pydantic import ValidationError

from naturalv2.outcome_metadata import OutcomeBounds


@pytest.mark.parametrize(
    "values",
    [
        {"minimum": 1, "maximum": 1},
        {"minimum": 2, "maximum": 1},
        {"minimum": float("nan"), "maximum": 1},
        {"minimum": 0, "maximum": float("inf")},
        {"minimum": 0, "maximum": 1, "unexpected": 2},
    ],
)
def test_outcome_bounds_require_a_finite_nonempty_interval(values):
    with pytest.raises(ValidationError):
        OutcomeBounds.model_validate(values)
