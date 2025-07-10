import numpy as np
import pandas as pd

from naturalv2.evals.experiment import Experiment
from naturalv2.pipeline import TREATMENT_COL_NAME
from naturalv2.utils import convert_enum_to_dicts, enumerate_strings


class NaturalOI:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.covariate_names = experiment.covariate_names
        self._num_treat = len(experiment.treatment_names)
        self._conditional_shape = [2]  # binary outcomes

    def _compute_outcome_conditionals(self, conditionals: pd.DataFrame) -> np.ndarray:
        options = enumerate_strings(
            {
                covariate: self.experiment.options[covariate]
                for covariate in self.experiment.covariate_names
            }
        )
        idx_to_feat = convert_enum_to_dicts(options, self.covariate_names)
        feat_dicts = [
            self.experiment.apply_transform(dct, repr_type="numeric")
            for dct in idx_to_feat
        ]

        outcome_conditionals = np.zeros((len(feat_dicts), self._num_treat))

        for i in range(len(feat_dicts)):
            features = feat_dicts[i]
            subset = conditionals.copy()
            # restrict posts using sampled features
            for key in self.covariate_names:
                subset = subset.loc[subset[key] == features[key]]
            for t in range(self._num_treat):
                subset_t = subset.loc[subset[TREATMENT_COL_NAME] == t]

                if len(subset_t) > 0:
                    py1_given_xt = np.array(
                        [
                            sum([j * prob[j] for j in range(len(prob))])
                            for prob in subset_t["y_given_tx_probs"]
                        ]
                    )
                    outcome_conditionals[i, t] = np.mean(py1_given_xt)

        return outcome_conditionals

    def get_individual_treatment_effects(
        self, conditionals: pd.DataFrame
    ) -> np.ndarray:
        # array of ITEs (treat2 - treat1) per unit corresponding to {outcome}
        conditionals = conditionals.copy()
        # outcome_idx = self.experiment.outcome_names.index(outcome)

        options = enumerate_strings(
            {
                covariate: self.experiment.options[covariate]
                for covariate in self.experiment.covariate_names
            }
        )
        idx_to_feat = convert_enum_to_dicts(options, self.covariate_names)
        feat_dicts = [
            self.experiment.apply_transform(dct, repr_type="numeric")
            for dct in idx_to_feat
        ]

        conditionals.loc[:, "y_given_tx_probs"] = conditionals.apply(
            lambda row: np.array(
                [float(prob) for prob in row["y_given_tx_probs"][1:-1].split()]
            ).reshape(self._conditional_shape),
            axis=1,
        )
        # choose probs corresponding to {outcome}
        # conditionals.loc[:, "y_given_tx_probs"] = conditionals.apply(
        #     lambda row: row["y_given_tx_probs"][2 * outcome_idx : 2 * (outcome_idx + 1)], axis=1
        # )

        self.outcome_conditionals = self._compute_outcome_conditionals(conditionals)
        all_ites = np.zeros((self._num_treat, len(conditionals)))
        for i, (_, row) in enumerate(conditionals.iterrows()):
            x = row[self.covariate_names].to_dict()
            x_idx = feat_dicts.index(x)
            for t in range(self._num_treat):
                all_ites[t, i] = self.outcome_conditionals[x_idx, t]

        return all_ites
