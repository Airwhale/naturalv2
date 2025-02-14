import ast
import re

def check_nonplacebo(intervention_names):
    nonplacebo_interventions = [name for name in intervention_names if "placebo" not in name.lower()]
    return len(nonplacebo_interventions) > 0

def check_noncontrol(type):
    if type == "NO_INTERVENTION":
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
    stats = {
        'total': 1,
        'randomized': 0,
        'multiple_noncontrol': 0,
        'nonhealthy': 0,
        'binary_endpoint': 0
    }
    if trial.alloc == "RANDOMIZED":
        stats['randomized'] = 1
        noncontrol_arms = [arm for arm in trial.arm_groups if check_noncontrol(arm.type)]
        nonplacebo_arms = [arm for arm in noncontrol_arms if check_nonplacebo(arm.intervention_names)]
        if len(nonplacebo_arms) >= 2:
            stats['multiple_noncontrol'] = 1
            if trial.inclusion_criteria.healthy_volunteers != "" and not trial.inclusion_criteria.healthy_volunteers:
                stats['nonhealthy'] = 1
                binary = False
                for endpoint in trial.primary_endpoints:
                    if check_binary_endpoint(endpoint.title):
                        stats['binary_endpoint'] = 1
                        binary = True
                        break
                if binary:
                    return stats, True
    return stats, False