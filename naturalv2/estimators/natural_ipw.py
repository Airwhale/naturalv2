import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from hydra.utils import instantiate

from naturalv2.utils import enumerate_strings, enum_to_dcts

class NaturalIPW:

    def __init__(self, experiment, name="natural_ipw"):
        self.name = name
        self.experiment = experiment
        self.covariate_names = experiment.covariate_names
        self.num_treat = len(experiment.treatment_names)
        self.num_out = len(experiment.outcome_names)
        self.conditional_shape = [self.num_treat, 2*self.num_out] # binary outcomes 
     
    def compute_prop_score(self, conditionals):
        options = enumerate_strings(self.experiment.get_options(self.covariate_names))
        idx_to_feat = enum_to_dcts(options, self.experiment.covariate_names)
        feat_dicts = [self.experiment.transform_samples(dct) for dct in idx_to_feat] 
        prop_score_lst = []
        
        for i in range(len(feat_dicts)):
            features = feat_dicts[i]
            subset = conditionals.copy()
            # restrict posts using sampled features
            for key in self.covariate_names:
                subset = subset.loc[subset[key] == features[key]]
            if len(subset) == 0:
                prop_scores = [0 for _ in range(self.num_treat)]
            else:
                # marginalize out Y
                propensity = subset[["probs"]].apply(lambda row: np.sum(row["probs"], axis=-1), axis=1) 
                # average over posts
                prop_scores = []
                for t in range(self.num_treat):
                    prop_t = propensity.apply(lambda arr: arr[t]).sum() / len(subset)
                    prop_scores.append(prop_t)        
            prop_score_lst.append(prop_scores)
        return prop_score_lst

    def get_ites(self, conditionals, outcome):
        # array of ITEs (treat2 - treat1) per unit
        conditionals = conditionals.copy()
        outcome_idx = self.experiment.outcome_names.index(outcome)
        options = enumerate_strings(self.experiment.get_options(self.covariate_names))
        idx_to_feat = enum_to_dcts(options, self.covariate_names)
        feat_dicts = [self.experiment.transform_samples(dct) for dct in idx_to_feat]

        conditionals.loc[:, "probs"] = conditionals.apply(
            lambda row: np.array(
                [float(prob) for prob in row["probs"][1:-1].split()]
            ).reshape(self.conditional_shape), axis=1
        )
        # choose probs corresponding to {outcome}
        conditionals.loc[:, "probs"] = conditionals.apply(
            lambda row: row["probs"][:, 2*outcome_idx: 2*(outcome_idx+1)], axis=1
        )
        
        self.prop_score_lst = self.compute_prop_score(conditionals)
        all_ites = np.zeros((self.num_treat, len(conditionals)))
        for i, row in enumerate(conditionals.iterrows()):
            row = row[1]
            probs = row["probs"]
            x = row[self.covariate_names].to_dict()
            # enumerate treatments
            for t in range(self.num_treat):
                # enumerate binary outcomes
                for y in range(2):
                    # probability of this enumerated possibility
                    posterior = probs[t, y]
                    # propensity score given x features
                    x_idx = feat_dicts.index(x)
                    z_given_x = self.prop_score_lst[x_idx][t]
                    # ignore propensity scores of 0
                    if z_given_x > 0:     
                        all_ites[t, i] = t * y * posterior / z_given_x                     
                        
        return all_ites