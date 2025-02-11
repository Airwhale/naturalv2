from ast import literal_eval
from naturalv2.utils import check_nonplacebo, check_binary_endpoint
from naturalv2.evals.clinicaltrials import ClinicalTrial

class Experiment:
    def __init__(self, nct_id, data_path, split="train"):
        trial = ClinicalTrial(data_path, nct_id)
        self.split = split
        self.trial_path = trial.data_path
        self.nct_id = trial.nct_id
        self.title = trial.official_title
        self.date = trial.estimated_completion if self.split=='test' else trial.results_first_posted
        self.references = [ref.get('citation', '') for ref in trial.references]

        self.outcome_treatment = []
        if split == 'test': # use enpdoints and arms to find outcome_treatment pairs
            for outcome in trial.primary_endpoints:
                for i, arm1 in enumerate(trial.arm_groups):
                    for j, arm2 in enumerate(trial.arm_groups):
                        if i < j:
                            self.outcome_treatment.append((outcome.title, (arm1.title, arm2.title)))
            
        else: # use result information to find outcome_treatment pairs
            outcome_results = [result for result in trial.outcome_results if check_binary_endpoint(result.title)]
            self.effect_sizes = []
            for result in outcome_results:
                result_cohorts = [cohort for cohort in result.cohorts if check_nonplacebo(cohort.title)]
                for i, cohort1 in enumerate(result_cohorts):
                    for j, cohort2 in enumerate(result_cohorts):
                        if i < j:  
                            effect_size = literal_eval(result.cohort_stats(cohort2)['value']) - literal_eval(result.cohort_stats(cohort1)['value'])
                            self.outcome_treatment.append((result.title, (cohort1.title, cohort2.title)))
                            self.effect_sizes.append(effect_size) # always cohort2 - cohort1
            
        self.covariates = [base.title for base in trial.baseline_char]
        self.inclusion_criteria = trial.inclusion_criteria.criteria

        # TODO: common names, misspellings
