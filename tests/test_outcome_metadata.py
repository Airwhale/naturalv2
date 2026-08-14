import pytest
from pydantic import ValidationError

from naturalv2.outcome_metadata import OutcomeBounds, infer_outcome_bounds


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Score range 1-49 with higher values signifying worse outcome", (1, 49)),
        ("Scores range from -10 to 10 points", (-10, 10)),
        ("Rated on a 0 to 55 scale", (0, 55)),
        ("Scale ranges from 0 (worst) to 100 (best)", (0, 100)),
        ("Score range 1\u201349", (1, 49)),
        ("Scale ranges from \u221210 to 10", (-10, 10)),
    ],
)
def test_infer_explicit_outcome_bounds(description, expected):
    bounds = infer_outcome_bounds(description)

    assert bounds is not None
    assert (bounds.minimum, bounds.maximum) == expected
    assert bounds.source == "description"


@pytest.mark.parametrize(
    "description",
    [
        "Each question is answered on a 6-point scale.",
        "Change from baseline to day 21",
        "Scores range from 1 to 7 per item and 7 to 49 for the total score",
        "Baseline scores ranged from 12 to 42 in enrolled patients.",
        "Observed scores ranged from 25 to 75 during screening.",
    ],
)
def test_ambiguous_or_non_range_text_is_not_parsed(description):
    assert infer_outcome_bounds(description) is None


@pytest.mark.parametrize(
    "values",
    [
        {"minimum": 1, "maximum": 1},
        {"minimum": 2, "maximum": 1},
        {"minimum": float("nan"), "maximum": 1},
        {"minimum": 0, "maximum": float("inf")},
    ],
)
def test_outcome_bounds_require_a_finite_nonempty_interval(values):
    with pytest.raises(ValidationError):
        OutcomeBounds.model_validate({**values, "source": "configured"})
