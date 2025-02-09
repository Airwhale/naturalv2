import ast
import re

def check_nonplacebo(title):
    title_lower = title.lower()
    if "placebo" in title_lower:
        return False
    return True

def check_binary_endpoint(text):
    BINARY_PATTERNS = [
    r"""
    \b(                  # Word boundary to ensure full-word match
        proportion       | # "proportion of ..."
        percentage       | # "percentage of ..."
        percent          | # "percent of ..."
        rate             | # "rate of ..."
        fraction           # "fraction of ..."
    )\s+of\s+              # Required "of" phrase with spaces
    (                      # Second group: Who the proportion applies to
        participants     | # "participants"
        subjects         | # "subjects"
        patients         | # "patients"
        individuals      | # "individuals"
        people           | # "people"
        volunteers       | # "volunteers"
        enrollees          # "enrollees"
    )\b                   # Ensure we match full words
    """
]
    return any(re.search(pattern, text, re.IGNORECASE | re.VERBOSE) for pattern in BINARY_PATTERNS)

def check_trial(trial):
    if trial.alloc == "RANDOMIZED":
        nonplacebo_interventions = [cohort.title for cohort in trial.outcome_results[0].cohorts if check_nonplacebo(cohort.title)]
        if len(nonplacebo_interventions) >= 2:
            if trial.inclusion_criteria.healthy_volunteers != "" and not trial.inclusion_criteria.healthy_volunteers:
                binary = False
                for result in trial.outcome_results:
                    if check_binary_endpoint(result.title):
                        binary = True
                        break
                if binary:
                    return True
    return False