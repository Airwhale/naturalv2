import pytest
from pydantic import ValidationError

from naturalv2.outcome_metadata import (
    OutcomeBounds,
    infer_outcome_bounds,
    is_change_from_baseline,
)


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


@pytest.mark.parametrize(
    "name,description",
    [
        (
            "Change from Baseline in Fatigue Severity Scale",
            "Change from baseline to week 12. Score range 1-49, higher is worse.",
        ),
        (
            "Mean change in FSS",
            "Mean change in the Fatigue Severity Scale. The scale ranges from 1 to 49.",
        ),
        (
            "Change in FUNCAP55 total score",
            "Change in total score. Possible scores range from 0 to 55.",
        ),
        (
            "Difference from Baseline in Fatigue Severity Scale",
            "Score range 1-49, higher is worse.",
        ),
        (
            "Reduction from baseline in fatigue score",
            "Score range 1-49, higher is worse.",
        ),
    ],
)
def test_change_endpoints_do_not_inherit_the_instrument_level_range(name, description):
    """A stated range describes the scale, not the span of a difference.

    Lithium reports -11.3 and -9.0 against a 1-49 instrument. Inferring [1, 49]
    for those endpoints would reject every improvement and keep every worsening.
    """
    assert infer_outcome_bounds(name, description) is None


def test_level_endpoints_still_infer_their_range():
    """The guard must not suppress inference for ordinary level endpoints."""
    bounds = infer_outcome_bounds(
        "Fatigue Severity Scale",
        "Total score at week 12. Score range 1-49, higher is worse.",
    )
    assert bounds is not None
    assert (bounds.minimum, bounds.maximum) == (1.0, 49.0)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Change from baseline in total score", True),
        ("Mean change in FSS at week 12", True),
        ("Percentage change from baseline", True),
        ("Difference from Baseline in FSS", True),
        ("Reduction from baseline in fatigue score", True),
        ("Delta FSS", True),
        ("Baseline to week 24 difference", True),
        ("Total score at week 12", False),
        ("Fatigue Severity Scale", False),
    ],
)
def test_change_from_baseline_detection(text, expected):
    assert is_change_from_baseline(text) is expected
