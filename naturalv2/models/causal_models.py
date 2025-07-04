from dataclasses import dataclass

import pandas as pd
from causallib.estimation import IPW, MarginalOutcomeEstimator, Standardization
from sklearn.linear_model import LinearRegression, LogisticRegression


@dataclass
class CausalData:
    """Data container for causal estimation."""

    X: pd.DataFrame
    T: pd.Series
    Y: pd.Series

    def __post_init__(self):
        """Validate data after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate data consistency."""
        if len(self.X) != len(self.T) or len(self.T) != len(self.Y):
            raise ValueError("X, T, and Y must have the same length")

        if self.X.isnull().any().any():
            raise ValueError("X contains missing values")

        if self.T.isnull().any() or self.Y.isnull().any():
            raise ValueError("T or Y contains missing values")


class DifferenceInMeans(object):
    def __init__(self):
        self.model_class = MarginalOutcomeEstimator
        self.learner = None
        self.fit_y = True
        self.outcome_y = True
        self.model = self.model_class(learner=self.learner)

    def fit(self, data: CausalData) -> None:
        if self.fit_y:
            self.model.fit(data.X, data.T, data.Y)
        else:
            self.model.fit(data.X, data.T)

    def estimate_individual_outcomes(self, data: CausalData) -> pd.Series:
        if self.outcome_y:
            outcomes = self.model.estimate_population_outcome(data.X, data.T, data.Y)
        else:
            outcomes = self.model.estimate_population_outcome(data.X, data.T)
        return self.model.estimate_effect(outcomes[1], outcomes[0])["diff"]


class IPSW(DifferenceInMeans):
    def __init__(self):
        super().__init__()
        self.model_class = IPW
        self.learner = LogisticRegression(solver="liblinear")
        self.fit_y = False
        self.outcome_y = True
        self.model = self.model_class(learner=self.learner)

    def estimate_individual_outcomes(self, data: CausalData) -> pd.Series:
        # ITE doesn't quite make sense for a MC version of IPW - we return y/P(T=t|x) for each unit.
        ipw_scores = self.model.compute_weights(data.X, data.T)
        ipw_scores *= len(data.X) / sum(ipw_scores)  # Hajek estmator
        return data.Y * ipw_scores


class OutcomeImputation(DifferenceInMeans):
    def __init__(self):
        super().__init__()
        self.model_class = Standardization
        self.learner = LinearRegression()
        self.fit_y = True
        self.outcome_y = False
        self.model = self.model_class(learner=self.learner)

    def estimate_individual_outcomes(self, data: CausalData) -> pd.Series:
        return self.model.estimate_individual_outcome(
            data.X, data.T, treatment_values=None
        )
