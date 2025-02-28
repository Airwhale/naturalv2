import numpy as np

from naturalv2.models.causal_models import DifferenceInMeans, IPSW, OutcomeImputation


class NaturalMC:

    def __init__(self, experiment, estimator_type="ipw"):
        self.experiment = experiment
        self.estimator_type = estimator_type
        self.covariate_names = experiment.covariate_names
        self.num_treat = len(experiment.treatment_names)
        self.num_out = len(experiment.outcome_names)

        self.causal_models = {
            "naive": DifferenceInMeans,
            "ipw": IPSW,
            "oi": OutcomeImputation,
        }
             
    def get_ites(self, conditionals, outcome):
        # array of ITEs (treat2 - treat1) per unit corresponding to {outcome}
        model = self.causal_models[self.estimator_type]()
        xs = conditionals[self.covariate_names].copy()
        ts = conditionals["treatment"].copy()
        ys = conditionals["outcome"].copy()
        data = (xs, ts, ys)
        model.fit(data)
        individual_outcomes = model.estimate_individual_outcomes(data)

        all_ites = np.zeros((self.num_treat, len(conditionals)))
        for t in range(self.num_treat):
            t_mask = [1 if treat == t else 0 for treat in ts]
            all_ites = individual_outcomes * t_mask 
        return all_ites
