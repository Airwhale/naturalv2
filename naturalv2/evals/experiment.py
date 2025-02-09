from ast import literal_eval
from naturalv2.utils import check_nonplacebo, check_binary_endpoint
from naturalv2.evals.clinicaltrials import ClinicalTrial

class Experiment:
    def __init__(self, nct_id, data_path, train=True):
        trial = ClinicalTrial(data_path, nct_id)
        self.trial_path = trial.data_path
        self.nct_id = trial.nct_id
        self.train = train

        outcome_results = [result for result in trial.outcome_results if check_binary_endpoint(result.title)]
        
        self.outcome_treatment, self.effect_sizes = [], []
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
