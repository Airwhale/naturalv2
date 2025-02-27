import numpy as np

from naturalv2.utils import enum_to_dcts, enumerate_strings


class NaturalOI:
    def __init__(self, experiment, name="natural_oi"):
        self.name = name
        self.experiment = experiment
        self.covariate_names = experiment.covariate_names
        self.num_treat = len(experiment.treatment_names)
        self.num_out = len(experiment.outcome_names)
        self.conditional_shape = [2 * self.num_out]  # binary outcomes

    def compute_outcome_cond(self, conditionals):
        options = enumerate_strings(self.experiment.get_options(self.covariate_names))
        idx_to_feat = enum_to_dcts(options, self.covariate_names)
        feat_dicts = [self.experiment.transform_samples(dct) for dct in idx_to_feat]

        outcome_conditionals = np.zeros((len(feat_dicts), self.num_treat))

        for i in range(len(feat_dicts)):
            features = feat_dicts[i]
            subset = conditionals.copy()
            # restrict posts using sampled features
            for key in self.covariate_names:
                subset = subset.loc[subset[key] == features[key]]
            for t in range(self.num_treat):
                subset_t = subset.loc[subset["treatment"] == t]

                if len(subset_t) > 0:
                    py1_given_xt = np.array([prob[1] for prob in subset_t["probs"]])
                    py1_given_xt = np.mean(py1_given_xt)
                    outcome_conditionals[i, t] = py1_given_xt

        return outcome_conditionals

    def get_ites(self, conditionals, outcome):
        # array of ITEs (treat2 - treat1) per unit corresponding to {outcome}
        conditionals = conditionals.copy()
        options = enumerate_strings(self.experiment.get_options(self.covariate_names))
        idx_to_feat = enum_to_dcts(options, self.covariate_names)
        feat_dicts = [self.experiment.transform_samples(dct) for dct in idx_to_feat]

        conditionals.loc[:, "probs"] = conditionals.apply(
            lambda row: np.array(
                [float(prob) for prob in row["probs"][1:-1].split()]
            ).reshape(self.conditional_shape),
            axis=1,
        )

        self.outcome_conditionals = self.compute_outcome_cond(conditionals)
        all_ites = np.zeros((self.num_treat, len(conditionals)))
        for i, (_, row) in enumerate(conditionals.iterrows()):
            x = row[self.covariate_names].to_dict()
            x_idx = feat_dicts.index(x)
            for t in range(self.num_treat):
                py1_given_xt = self.outcome_conditionals[x_idx, t]
                all_ites[t, i] = py1_given_xt

        return all_ites
