from ast import literal_eval

from naturalv2.evals.clinicaltrials import ClinicalTrial
from naturalv2.utils import check_binary_endpoint, check_noncontrol, check_nonplacebo


class Experiment:
    def __init__(self, nct_id, data_path, split="train"):
        trial = ClinicalTrial(data_path, nct_id)
        self.split = split
        self.trial_path = trial.data_path
        self.nct_id = trial.nct_id
        self.title = trial.official_title
        self.date = (
            trial.estimated_completion
            if self.split == "test"
            else trial.results_first_posted
        )
        self.references = [ref.get("citation", "") for ref in trial.references]

        self.outcome_treatment = []
        if split == "test":  # use enpdoints and arms to find outcome_treatment pairs
            outcomes = [
                outcome
                for outcome in trial.primary_endpoints
                if check_binary_endpoint(outcome.title)
            ]
            arms = [
                arm
                for arm in trial.arm_groups
                if check_noncontrol(arm.type)
                and check_nonplacebo(arm.intervention_names)
            ]
            for outcome in outcomes:
                for i, arm1 in enumerate(arms):
                    for j, arm2 in enumerate(arms):
                        if i < j:
                            self.outcome_treatment.append(
                                (outcome.title, (arm1.title, arm2.title))
                            )

        else:  # use result information to find outcome_treatment pairs
            outcomes = [
                result
                for result in trial.outcome_results
                if check_binary_endpoint(result.title)
            ]
            self.effect_sizes = []
            for result in outcomes:
                arms = [
                    cohort
                    for cohort in result.cohorts
                    if check_nonplacebo([cohort.title])
                ]
                for i, cohort1 in enumerate(arms):
                    for j, cohort2 in enumerate(arms):
                        if i < j:
                            effect_size = literal_eval(
                                result.cohort_stats(cohort2)["value"]
                            ) - literal_eval(result.cohort_stats(cohort1)["value"])
                            self.outcome_treatment.append(
                                (result.title, (cohort1.title, cohort2.title))
                            )
                            self.effect_sizes.append(
                                effect_size / 100
                            )  # always cohort2 - cohort1

        self.outcome_names = [out.title for out in outcomes]
        self.treatment_names = [arm.title for arm in arms]
        self.covariate_names = [base.title for base in trial.baseline_char]
        self.inclusion_criteria = trial.inclusion_criteria.criteria

    def download_data(self):
        self.data_dump = []  # list of paths to relevant data dumps, one per source

    def curate_data(self):
        self.curated_data_path = ""  # path to curated data

        # TODO: common names, misspellings
