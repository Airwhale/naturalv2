from ast import literal_eval

from naturalv2.evals.clinicaltrials import ClinicalTrial
from naturalv2.utils import check_binary_endpoint, check_noncontrol, check_nonplacebo


class Experiment:
    def __init__(self, nct_id, data_path, split="train"):
        self.trial = ClinicalTrial(data_path, nct_id)
        self.split = split
        self.trial_path = self.trial.data_path
        self.nct_id = self.trial.nct_id
        self.title = self.trial.official_title
        self.date = (
            self.trial.estimated_completion
            if self.split == "test"
            else self.trial.results_first_posted
        )
        self.references = [ref.get("citation", "") for ref in self.trial.references]

        self.set_outcome_treatment_effects()
        self.covariate_names = [base.title for base in self.trial.baseline_char]
        self.inclusion_criteria = self.trial.inclusion_criteria.criteria

        self.data_dump = []  # list of paths to relevant data dumps, one per source
        self.curated_data_path = ""  # path to curated data
        self.treatment_common_names = []
        self.outcome_common_names = []
        self.extended_covariate_names = []  # inclusion-related binary variables
        self.options = {}
        self.question_prompts = {}

        self.set_transforms()

    def set_outcome_treatment_effects(self):
        self.outcome_treatment = []
        if (
            self.split == "test"
        ):  # use enpdoints and arms to find outcome_treatment pairs
            outcomes = [
                outcome
                for outcome in self.trial.primary_endpoints
                if check_binary_endpoint(outcome.title)
            ]
            arms = [
                arm
                for arm in self.trial.arm_groups
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
                for result in self.trial.outcome_results
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
                            effect1, effect2 = (
                                result.cohort_stats(cohort1),
                                result.cohort_stats(cohort2),
                            )
                            denom1, denom2 = (
                                literal_eval(cohort1.denom),
                                literal_eval(cohort2.denom),
                            )
                            if (
                                effect1 is not None
                                and effect2 is not None
                                and denom1 > 0
                                and denom2 > 0
                            ):
                                effect1 = literal_eval(effect1["value"])
                                effect2 = literal_eval(effect2["value"])
                            else:
                                continue

                            # divide by cohort size or 100 if result is a percentage
                            effect1 = (
                                effect1 / 100
                                if "percent" in result.unit_of_measure.lower()
                                else effect1 / denom1
                            )
                            effect2 = (
                                effect2 / 100
                                if "percent" in result.unit_of_measure.lower()
                                else effect2 / denom2
                            )
                            effect_size = effect2 - effect1
                            self.outcome_treatment.append(
                                (result.title, (cohort1.title, cohort2.title))
                            )
                            self.effect_sizes.append(
                                effect_size
                            )  # always cohort2 - cohort1

        self.outcome_names = [out.title for out in outcomes]
        self.treatment_names = [arm.title for arm in arms]

    def hard_filter_ty(self, extractions):
        for name in self.treatment_names + self.outcome_names:
            extractions = extractions[extractions[name].isin(self.options[name])]
        return extractions

    def hard_filter_inclusion(self, extractions):
        for name in self.extended_covariate_names:
            extractions = extractions[
                extractions[name].lower().isin(["yes", "unknown"])
            ]
        return extractions

    def discretize(self, extractions):
        return

    def set_transforms(self):
        self.numerical_repr = {}
        self.language_repr = {}
