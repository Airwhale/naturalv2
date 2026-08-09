import numpy as np
import pandas as pd
import pytest

from naturalv2.estimators.natural_ipw import NaturalIPW
from naturalv2.estimators.natural_oi import NaturalOI
from naturalv2.pipeline import TREATMENT_COL_NAME


class FakeExperiment:
    covariate_names = ["cov"]
    treatment_names = ["A", "B"]
    options = {"cov": ["X", "Y"]}

    def apply_transform(self, d, repr_type="numeric"):
        m = {"X": 0, "Y": 1}
        return {k: m[v] for k, v in d.items()}


def test_ipw_computes_horvitz_thompson_estimate():
    # cov=X: P(T=A,Y=Yes)=1 -> A row gets full credit, B gets 0 (P(T=B|X)=0, skipped).
    # cov=Y: P(T=B,Y=Yes)=1 -> symmetric.
    df = pd.DataFrame(
        {
            "cov_discretized": [0, 0, 1, 1],
            "ty_given_x_probs": ["[0,1,0,0]", "[0,1,0,0]", "[0,0,0,1]", "[0,0,0,1]"],
        }
    )
    out = NaturalIPW(FakeExperiment()).get_individual_treatment_effects(df)
    assert (out == [[1, 1, 0, 0], [0, 0, 1, 1]]).all()


def test_oi_predicts_potential_outcome_regardless_of_assigned_treatment():
    df = pd.DataFrame(
        {
            "cov_discretized": [0, 0, 1, 1],
            f"{TREATMENT_COL_NAME}_discretized": [0, 1, 0, 1],
            "y_given_tx_probs": ["[0,1]", "[1,0]", "[0.5,0.5]", "[0,1]"],
        }
    )
    out = NaturalOI(FakeExperiment()).get_individual_treatment_effects(df)
    assert (out == [[1, 1, 0.5, 0.5], [0, 0, 1, 1]]).all()


def test_oi_zero_support_combo_is_nan():
    # No row has cov=Y with T=B, so that (x, t) cell has no data. Unlike
    # NaturalMC's "oi" (a fitted regression that can extrapolate), this
    # estimator only averages real rows per combo, so it's undefined.
    df = pd.DataFrame(
        {
            "cov_discretized": [0, 0, 1, 1],
            f"{TREATMENT_COL_NAME}_discretized": [0, 1, 0, 0],
            "y_given_tx_probs": ["[0,1]", "[1,0]", "[0.5,0.5]", "[0.5,0.5]"],
        }
    )
    out = NaturalOI(FakeExperiment()).get_individual_treatment_effects(df)
    assert np.isnan(out[1, 2:]).all()


def test_ipw_unseen_covariate_value_raises():
    class RestrictedOptions(FakeExperiment):
        options = {"cov": ["X"]}  # 'Y' unseen

    df = pd.DataFrame({"cov_discretized": [0, 1], "ty_given_x_probs": ["[0,1,0,0]", "[0,0,0,1]"]})
    with pytest.raises(ValueError, match="not in list"):
        NaturalIPW(RestrictedOptions()).get_individual_treatment_effects(df)
