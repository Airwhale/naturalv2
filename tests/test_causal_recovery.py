"""Recovery tests: given a known confounded data-generating process, do the
estimators recover the true effect, and better than a naive comparison would?

DGP shared across tests: covariate X (risk: 0=low, 1=high) confounds
treatment assignment (P(T=1|X=0)=0.2, P(T=1|X=1)=0.8 -- "confounding by
indication") and the outcome (P(Y=1|X=0,*)=0.6/0.8, P(Y=1|X=1,*)=0.2/0.4).
The true additive effect of T=1 vs T=0 is +0.2 in both strata, so the true
population ATE is +0.2 -- but a naive, unadjusted comparison is biased
because high-risk patients get treatment 1 far more often.
"""

import numpy as np
import pandas as pd
import pytest

from naturalv2.cli.estimate_ate import _calculate_treatment_responses
from naturalv2.estimators.natural_ipw import NaturalIPW
from naturalv2.estimators.natural_mc import NaturalMC
from naturalv2.estimators.natural_oi import NaturalOI
from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME


TRUE_ATE = 0.2


class FakeExperiment:
    covariate_names = ["risk"]
    treatment_names = ["A", "B"]
    options = {"risk": ["Low", "High"]}
    status = "not_completed"
    apo_outcome_treatment = []

    def apply_transform(self, d, repr_type="numeric"):
        return {k: {"Low": 0, "High": 1}[v] for k, v in d.items()}

    def is_binary_outcome(self, outcome):
        return True


def simulate_xty(n, seed):
    """Raw (X, T, Y) draws from the DGP -- the "ground truth" reports."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=n)
    t = (rng.random(n) < np.where(x == 0, 0.2, 0.8)).astype(int)
    p_y = np.select(
        [(x == 0) & (t == 0), (x == 0) & (t == 1), (x == 1) & (t == 0), (x == 1) & (t == 1)],
        [0.6, 0.8, 0.2, 0.4],
    )
    y = (rng.random(n) < p_y).astype(int)
    return pd.DataFrame({"x": x, "t": t, "y": y})


def to_sampled_columns(sim):
    """As if perfectly sampled by SampleTYStage: one row per individual."""
    return pd.DataFrame(
        {
            "risk_discretized": sim["x"],
            f"{TREATMENT_COL_NAME}_discretized": sim["t"],
            f"{OUTCOME_COL_NAME}_discretized": sim["y"].astype(float),
        }
    )


def naive_diff(sim):
    return sim.loc[sim["t"] == 1, "y"].mean() - sim.loc[sim["t"] == 0, "y"].mean()


def test_ipw_recovers_ate_from_exact_conditionals():
    # One row per risk stratum, carrying the exact P(T,Y|X) joint -- as if the
    # LLM extracted it perfectly. Order per row: [T0Y0, T0Y1, T1Y0, T1Y1].
    df = pd.DataFrame(
        {
            "risk_discretized": [0, 1],
            "ty_given_x_probs": ["[0.32,0.48,0.04,0.16]", "[0.16,0.04,0.48,0.32]"],
        }
    )
    apo = NaturalIPW(FakeExperiment()).get_individual_treatment_effects(df).mean(axis=1)
    assert apo == pytest.approx([0.4, 0.6])
    assert apo[1] - apo[0] == pytest.approx(TRUE_ATE)


def test_oi_recovers_ate_from_exact_conditionals():
    # One row per (risk, treatment) cell, carrying the exact P(Y=1|X,T).
    df = pd.DataFrame(
        {
            "risk_discretized": [0, 0, 1, 1],
            f"{TREATMENT_COL_NAME}_discretized": [0, 1, 0, 1],
            "y_given_tx_probs": ["[0.4,0.6]", "[0.2,0.8]", "[0.8,0.2]", "[0.6,0.4]"],
        }
    )
    apo = NaturalOI(FakeExperiment()).get_individual_treatment_effects(df).mean(axis=1)
    assert apo == pytest.approx([0.4, 0.6])
    assert apo[1] - apo[0] == pytest.approx(TRUE_ATE)


def test_ipw_corrects_confounding_bias_from_estimated_conditionals():
    # Realistic version: ty_given_x_probs is *estimated* per stratum from a
    # finite sample (like an LLM would from many similar reports), not the
    # exact analytic joint -- so this also carries sampling noise.
    sim = simulate_xty(n=4000, seed=1)
    rows = []
    for x_val, group in sim.groupby("x"):
        joint = [
            float(((group["t"] == tv) & (group["y"] == yv)).sum()) / len(group)
            for tv in (0, 1)
            for yv in (0, 1)
        ]
        rows += [(int(x_val), str(joint))] * len(group)
    df = pd.DataFrame(rows, columns=["risk_discretized", "ty_given_x_probs"])

    apo = NaturalIPW(FakeExperiment()).get_individual_treatment_effects(df).mean(axis=1)
    corrected_ate = apo[1] - apo[0]
    bias = naive_diff(sim)
    assert abs(bias - TRUE_ATE) > 0.15
    assert abs(corrected_ate - TRUE_ATE) < 0.1
    assert abs(corrected_ate - TRUE_ATE) < abs(bias - TRUE_ATE)


def test_oi_corrects_confounding_bias_from_estimated_conditionals():
    sim = simulate_xty(n=4000, seed=1)
    rows = []
    for (x_val, t_val), group in sim.groupby(["x", "t"]):
        p_yes = float(group["y"].mean())
        rows += [(int(x_val), int(t_val), str([1 - p_yes, p_yes]))] * len(group)
    df = pd.DataFrame(
        rows, columns=["risk_discretized", f"{TREATMENT_COL_NAME}_discretized", "y_given_tx_probs"]
    )

    apo = NaturalOI(FakeExperiment()).get_individual_treatment_effects(df).mean(axis=1)
    corrected_ate = apo[1] - apo[0]
    bias = naive_diff(sim)
    assert abs(bias - TRUE_ATE) > 0.15
    assert abs(corrected_ate - TRUE_ATE) < 0.1
    assert abs(corrected_ate - TRUE_ATE) < abs(bias - TRUE_ATE)


@pytest.mark.parametrize("estimator_type", ["ipw", "oi"])
def test_natural_mc_corrects_confounding_bias(estimator_type):
    sim = simulate_xty(n=3000, seed=42)
    df = to_sampled_columns(sim)
    bias = naive_diff(sim)
    assert abs(bias - TRUE_ATE) > 0.15  # confounding badly biases the naive comparison

    estimator = NaturalMC(FakeExperiment(), estimator_type=estimator_type)
    _, weighted_responses = _calculate_treatment_responses(
        FakeExperiment(), "dummy_outcome", estimator, df,
        bootstrap_size=10, seed=0, use_inclusion_weights=False,
    )
    corrected_ate = weighted_responses[1] - weighted_responses[0]
    assert abs(corrected_ate - TRUE_ATE) < 0.15
    assert abs(corrected_ate - TRUE_ATE) < abs(bias - TRUE_ATE)
