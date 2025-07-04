from typing import Literal

import numpy as np
import pandas as pd

from naturalv2.evals.experiment import Experiment
from naturalv2.models.causal_models import (
    IPSW,
    CausalData,
    DifferenceInMeans,
    OutcomeImputation,
)
from naturalv2.pipeline import TREATMENT_COL_NAME


class NaturalMC:
    def __init__(
        self, experiment: Experiment, estimator_type: Literal["ipw", "oi"] = "ipw"
    ) -> None:
        self.experiment = experiment
        self.estimator_type = estimator_type
        self.num_treat = len(experiment.treatment_names)
        # self.num_out = len(experiment.outcome_names)

        self.causal_models: dict[
            str, type[DifferenceInMeans] | type[IPSW] | type[OutcomeImputation]
        ] = {
            "naive": DifferenceInMeans,
            "ipw": IPSW,
            "oi": OutcomeImputation,
        }

    def get_individual_treatment_effects(
        self, conditionals: pd.DataFrame, outcome: str
    ) -> np.ndarray:
        # array of ITEs (treat2 - treat1) per unit corresponding to {outcome}
        model = self.causal_models[self.estimator_type]()

        data = CausalData(
            X=conditionals[self.experiment.covariate_names].copy(),  # covariates
            T=conditionals[TREATMENT_COL_NAME].copy(),  # treatment
            Y=conditionals[outcome].copy(),  # outcome
        )

        model.fit(data)
        individual_outcomes = model.estimate_individual_outcomes(data)

        all_ites = np.zeros((self.num_treat, len(conditionals)))
        for t in range(self.num_treat):
            if self.estimator_type == "ipw":
                t_mask = [1 if treat == t else 0 for treat in data.T]
                all_ites[t, :] = (individual_outcomes * t_mask).to_numpy()
            elif self.estimator_type == "oi":
                all_ites[t, :] = individual_outcomes[t].to_numpy()
            else:
                raise NotImplementedError(
                    f"Estimator type '{self.estimator_type}' not implemented."
                )
        return all_ites
