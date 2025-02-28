from sklearn.linear_model import LogisticRegression, LinearRegression
from causallib.estimation import MarginalOutcomeEstimator, IPW, Standardization, 


class DifferenceInMeans(object):
    
    def __init__(self):
        self.model_class = MarginalOutcomeEstimator
        self.learner = None
        self.fit_y = True
        self.outcome_y = True
        self.model = self.model_class(learner=self.learner)

    def fit(self, data):
        X, T, Y = data
        if self.fit_y:
            self.model.fit(X, T, Y)
        else:
            self.model.fit(X, T)

    def get_effect(self, data):
        X, T, Y = data
        if self.outcome_y:
            outcomes = self.model.estimate_population_outcome(X, T, Y)
        else:
            outcomes = self.model.estimate_population_outcome(X, T)
        effect = self.model.estimate_effect(outcomes[1], outcomes[0])["diff"]
        return effect
    

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
        X, T, Y = data 
        ipw_scores = self.model.compute_weights(X, T) 
        ipw_scores *= len(X) / sum(ipw_scores) # Hajek estmator
        res = Y * ipw_score
        return res


class OutcomeImputation(DifferenceInMeans):
    
    def __init__(self):
        super().__init__()
        self.model_class = Standardization
        self.learner = LinearRegression()
        self.fit_y = True
        self.outcome_y = False
        self.model = self.model_class(learner=self.learner)

    def estimate_individual_outcomes(self, data):
        X, T, _ = data
        individual_outcomes = self.model.estimate_individual_outcome(X, T, treatment_values=None)
        return individual_outcomes
