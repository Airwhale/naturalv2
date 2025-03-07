from causallib.estimation import IPW, MarginalOutcomeEstimator, Standardization
from sklearn.linear_model import LinearRegression, LogisticRegression


class DifferenceInMeans(object):
    def __init__(self):
        self.model_class = MarginalOutcomeEstimator
        self.learner = None
        self.fit_y = True
        self.outcome_y = True
        self.model = self.model_class(learner=self.learner)

    def fit(self, data):
        xs, ts, ys = data
        if self.fit_y:
            self.model.fit(xs, ts, ys)
        else:
            self.model.fit(xs, ts)

    def get_effect(self, data):
        xs, ts, ys = data
        if self.outcome_y:
            outcomes = self.model.estimate_population_outcome(xs, ts, ys)
        else:
            outcomes = self.model.estimate_population_outcome(xs, ts)
        return self.model.estimate_effect(outcomes[1], outcomes[0])["diff"]


class IPSW(DifferenceInMeans):
    def __init__(self):
        super().__init__()
        self.model_class = IPW
        self.learner = LogisticRegression(solver="liblinear")
        self.fit_y = False
        self.outcome_y = True
        self.model = self.model_class(learner=self.learner)

    def estimate_individual_outcomes(self, data):
        # ITE doesn't quite make sense for a MC version of IPW - we return y/P(T=t|x) for each unit.
        xs, ts, ys = data
        ipw_scores = self.model.compute_weights(xs, ts)
        ipw_scores *= len(xs) / sum(ipw_scores)  # Hajek estmator
        return ys * ipw_scores


class OutcomeImputation(DifferenceInMeans):
    def __init__(self):
        super().__init__()
        self.model_class = Standardization
        self.learner = LinearRegression()
        self.fit_y = True
        self.outcome_y = False
        self.model = self.model_class(learner=self.learner)

    def estimate_individual_outcomes(self, data):
        xs, ts, _ = data
        return self.model.estimate_individual_outcome(xs, ts, treatment_values=None)
