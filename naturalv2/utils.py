import ast

def check_nonplacebo(title):
    title_lower = title.lower()
    if "placebo" in title_lower:
        return False
    return True

def check_trial(trial):
    if trial.alloc == "RANDOMIZED":
        nonplacebo_interventions = [i.title for i in trial.interventions if check_nonplacebo(i.title)]
        if len(nonplacebo_interventions) >= 2:
            if trial.inclusion_criteria.healthy_volunteers != "" and not trial.inclusion_criteria.healthy_volunteers:
                return True
    return False

