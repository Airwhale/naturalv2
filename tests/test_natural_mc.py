import numpy as np
import pandas as pd
import pytest

from naturalv2.estimators.natural_mc import NaturalMC
from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME


class FakeExperiment:
    covariate_names = ["cov"]
    treatment_names = ["A", "B", "C"]


def make_data(n, treatments, seed=0):
    """``treatments``: list of treatment indices assigned per row (len n)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cov_discretized": rng.integers(0, 2, size=n),
            f"{TREATMENT_COL_NAME}_discretized": treatments,
            f"{OUTCOME_COL_NAME}_discretized": rng.normal(size=n),
        }
    )


@pytest.mark.parametrize("estimator_type", ["ipw", "oi"])
def test_two_arms_supported(estimator_type):
    data = make_data(20, [0, 1] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type=estimator_type)
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert ites.shape == (3, 20)
    assert not np.isnan(ites[:2, :]).any()  # treatment 'C' has no data, row 2 is 0/NaN


@pytest.mark.parametrize("estimator_type", ["ipw", "oi"])
def test_three_arms_supported(estimator_type):
    data = make_data(30, [0, 1, 2] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type=estimator_type)
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert not np.isnan(ites).any()


def test_ipw_missing_treatment_is_zero_not_nan():
    # Caller (`_calculate_treatment_responses`) is responsible for turning this
    # into NaN once it knows the treatment is genuinely unsupported.
    data = make_data(20, [0, 1] * 10)  # treatment 'C' (index 2) never appears
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert (ites[2, :] == 0).all()


def test_ipw_does_not_dilute_treatment_outcomes_with_other_arms():
    data = pd.DataFrame(
        {
            "cov_discretized": [0, 1] * 10,
            f"{TREATMENT_COL_NAME}_discretized": [0] * 10 + [1] * 10,
            f"{OUTCOME_COL_NAME}_discretized": [2.0] * 10 + [4.0] * 10,
        }
    )
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")

    average_outcomes = mc.get_individual_treatment_effects(data, outcome="dummy").mean(
        axis=1
    )

    assert average_outcomes[:2] == pytest.approx([2.0, 4.0])


def test_ipw_outcome_is_unchanged_when_other_arm_grows():
    data = pd.DataFrame(
        {
            "cov_discretized": [0, 1] * 4,
            f"{TREATMENT_COL_NAME}_discretized": [0] * 4 + [1] * 4,
            f"{OUTCOME_COL_NAME}_discretized": [2.0] * 4 + [4.0] * 4,
        }
    )
    additional_other_arm_rows = pd.DataFrame(
        {
            "cov_discretized": [0, 1] * 8,
            f"{TREATMENT_COL_NAME}_discretized": [1] * 16,
            f"{OUTCOME_COL_NAME}_discretized": [4.0] * 16,
        }
    )
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")

    original_outcome = mc.get_individual_treatment_effects(data, outcome="dummy")[
        0
    ].mean()
    expanded_outcome = mc.get_individual_treatment_effects(
        pd.concat([data, additional_other_arm_rows], ignore_index=True),
        outcome="dummy",
    )[0].mean()

    assert original_outcome == pytest.approx(2.0)
    assert expanded_outcome == pytest.approx(original_outcome)


def test_oi_missing_treatment_extrapolates_without_crash():
    data = make_data(20, [0, 1] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type="oi")
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert ites.shape == (3, 20)
    assert not np.isnan(ites).any()


def test_ipw_fit_error_propagates_with_context():
    # LogisticRegression needs >=2 classes to fit at all.
    data = make_data(10, [0] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")

    with pytest.raises(ValueError) as exc_info:
        mc.get_individual_treatment_effects(data, outcome="dummy")

    assert "outcome='dummy'" in " ".join(exc_info.value.__notes__)


def test_oi_single_treatment_value_still_returns_finite_values():
    # LinearRegression has no such requirement -- it just extrapolates.
    data = make_data(10, [0] * 10)
    mc = NaturalMC(FakeExperiment(), estimator_type="oi")
    ites = mc.get_individual_treatment_effects(data, outcome="dummy")
    assert not np.isnan(ites).any()


def test_missing_required_column_raises():
    data = make_data(5, [0, 1, 0, 1, 0]).drop(
        columns=[f"{TREATMENT_COL_NAME}_discretized"]
    )
    mc = NaturalMC(FakeExperiment(), estimator_type="ipw")
    with pytest.raises(ValueError, match="discretized"):
        mc.get_individual_treatment_effects(data, outcome="dummy")


@pytest.mark.xfail(
    reason=(
        "The per-arm Hajek normaliser assumes the caller takes an unweighted "
        "mean. estimate_ate weights by inclusion probability, so the normaliser "
        "is wrong for every non-uniform inclusion vector. Fixing it needs the "
        "normalisation moved to the caller, which is the only place that knows "
        "those probabilities -- a contract change, tracked separately."
    ),
    strict=True,
)
def test_ipw_arms_survive_non_uniform_inclusion_weights():
    """Two balanced arms should read 2 and 4 whoever is judged eligible."""
    n_per_arm = 10
    treatments = np.array([0] * n_per_arm + [1] * n_per_arm)
    outcomes = np.array([2.0] * n_per_arm + [4.0] * n_per_arm)
    propensity = np.ones(2 * n_per_arm)
    # Arm 0 reads as less trial-eligible than arm 1 -- an ordinary outcome of
    # the inclusion_prob stage, not a pathological input.
    inclusion = np.array([0.2] * n_per_arm + [0.9] * n_per_arm)

    scores = pd.Series(propensity)
    scores = (
        scores
        * len(treatments)
        / scores.groupby(pd.Series(treatments)).transform("sum")
    )
    ites = pd.Series(outcomes) * scores
    responses = np.array(
        [
            [ites[i] if treatments[i] == arm else 0.0 for i in range(len(treatments))]
            for arm in (0, 1)
        ]
    )

    estimated = np.average(responses, axis=1, weights=inclusion)
    np.testing.assert_allclose(estimated, [2.0, 4.0])
