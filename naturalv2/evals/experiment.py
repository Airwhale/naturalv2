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
        
        self.outcomes = [result.title for result in outcome_results]
        self.treatments, self.ground_truths = {}, {}
        for result in outcome_results:
            result_cohorts = [cohort for cohort in result.cohorts if check_nonplacebo(cohort.title)]
            self.treatments[result.title] = [cohort.title for cohort in result_cohorts]
            gt = []
            for i, cohort1 in enumerate(result_cohorts):
                for j, cohort2 in enumerate(result_cohorts):
                    if i < j:  
                        effect_size = literal_eval(result.cohort_stats(cohort1)['value']) - literal_eval(result.cohort_stats(cohort2)['value'])
                        gt.append(effect_size)
            self.ground_truths[result.title] = gt
            
        self.covariates = [base.title for base in trial.baseline_char]
        self.inclusion_criteria = trial.inclusion_criteria.criteria
