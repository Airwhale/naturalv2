import re
from typing import Literal, Union

from pydantic import BaseModel


class TYFilterResponse(BaseModel):
    """Response format for treatment-outcome filter stage"""

    start_weight: Union[float, Literal["Unknown"]]
    end_weight: Union[float, Literal["Unknown"]]
    weight_unit: Literal["kg", "lb", "Unknown"]
    weight_change: Union[float, Literal["Unknown"]]
    percentage_weight_change: Union[float, Literal["Unknown"]]
    treatment: Literal["Semaglutide", "Tirzepatide", "Other", "Unknown"]


class KnownsResponse(BaseModel):
    t2dm: Literal["Yes", "No", "Unknown"]
    metformin: Literal["Yes", "No", "Unknown"]
    bmi: Union[float, Literal["Unknown"]]
    age: Union[int, Literal["Unknown"]]
    sex: Literal["Male", "Female", "Unknown"]
    start_HbA1c: Union[float, Literal["Unknown"]]  # noqa: N815
    end_HbA1c: Union[float, Literal["Unknown"]]  # noqa: N815
    country: Union[str, Literal["Unknown"]]
    start_weight: Union[float, Literal["Unknown"]]
    end_weight: Union[float, Literal["Unknown"]]
    weight_unit: Literal["kg", "lb", "Unknown"]
    weight_change: Union[float, Literal["Unknown"]]
    percentage_weight_change: Union[float, Literal["Unknown"]]
    duration_days: Union[int, Literal["Unknown"]]
    treatment: Literal[
        "Semaglutide",
        "Tirzepatide",
        "Ozempic",
        "Wegovy",
        "Rybelsus",
        "Mounjaro",
        "Zepbound",
        "Unknown",
    ]
    dosage: Union[float, Literal["Unknown"]]
    target_achieved: Literal["Yes", "No", "Unknown"]


class ImputationsResponse(BaseModel):
    bmi: Union[float, Literal["Unknown"]]
    age: Union[int, Literal["Unknown"]]
    sex: Literal["Male", "Female", "Unknown"]
    start_HbA1c: Union[float, Literal["Unknown"]]  # noqa: N815
    country: Union[str, Literal["Unknown"]]
    start_weight: Union[float, Literal["Unknown"]]
    duration_days: Union[int, Literal["Unknown"]]


def check_nonplacebo(intervention_names):
    nonplacebo_interventions = [
        name for name in intervention_names if "placebo" not in name.lower()
    ]
    return len(nonplacebo_interventions) > 0


def check_noncontrol(intervention_type):
    return intervention_type != "NO_INTERVENTION"


def check_binary_endpoint(text):
    binary_patterns = [
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
    return any(
        re.search(pattern, text, re.IGNORECASE | re.VERBOSE)
        for pattern in binary_patterns
    )


# Rest of the code remains unchanged below this line
def check_trial(trial):
    stats = {
        "total": 1,
        "randomized": 0,
        "multiple_noncontrol": 0,
        "nonhealthy": 0,
        "binary_endpoint": 0,
    }
    if trial.alloc == "RANDOMIZED":
        stats["randomized"] = 1
        noncontrol_arms = [
            arm for arm in trial.arm_groups if check_noncontrol(arm.type)
        ]
        nonplacebo_arms = [
            arm for arm in noncontrol_arms if check_nonplacebo(arm.intervention_names)
        ]
        if len(nonplacebo_arms) >= 2:
            stats["multiple_noncontrol"] = 1
            if (
                trial.inclusion_criteria.healthy_volunteers != ""
                and not trial.inclusion_criteria.healthy_volunteers
            ):
                stats["nonhealthy"] = 1
                binary = False
                for endpoint in trial.primary_endpoints:
                    if check_binary_endpoint(endpoint.title):
                        stats["binary_endpoint"] = 1
                        binary = True
                        break
                if binary:
                    return stats, True
    return stats, False


def qa_interleaved_enum(q_dct, options_dct, a_enum, to_enum):
    all_interleaved_options = []
    alph = ["a) ", "b) ", "c) ", "d) "]
    for option in a_enum:
        interleaved_enum = " \n\n## Questions"
        for num in range(len(to_enum)):
            key = to_enum[num]
            interleaved_enum += " \n\nQ: " + q_dct[key]
            interleaved_enum += " \nOptions: "
            for i in range(len(options_dct[key])):
                interleaved_enum += alph[i] + options_dct[key][i] + " "
            split_option = [i.split(":") for i in option.split(",")]
            interleaved_enum += " \nA: " + split_option[num][1][1:]
        all_interleaved_options.append(interleaved_enum)
    return all_interleaved_options


def concatenate_q(dct):
    keys = list(dct.keys())
    num = 1
    all_qs = " \nAnswer the following questions."
    for key in keys:
        all_qs += " \nQ" + str(num) + ": " + dct[key]
        num += 1
    all_qs += "\n"
    return all_qs


def enumerate_strings(dct, string=True):
    keys = list(dct.keys())
    keys.reverse()
    num = len(keys)
    all_enumerated = dct[keys[0]]
    all_enumerated = ["A" + str(num) + ": " + e for e in all_enumerated]
    for key in keys[1:]:
        num -= 1
        cur_len = len(all_enumerated)
        all_enumerated *= len(dct[key])
        for j in range(len(dct[key])):
            all_enumerated[j * cur_len : (j + 1) * cur_len] = [
                dct[key][j] + ", " + e
                for e in all_enumerated[j * cur_len : (j + 1) * cur_len]
            ]
        all_enumerated = ["A" + str(num) + ": " + e for e in all_enumerated]
    return all_enumerated


def enum_to_dcts(enumerated, to_enum):
    return_dcts = []
    for elem in enumerated:
        separate = [i.split(":") for i in elem.split(",")]
        dct = {}
        for field in range(len(to_enum)):
            dct[to_enum[field]] = separate[field][1][1:]
        return_dcts.append(dct)
    return return_dcts


def get_sample_text(dct, dataset):
    all_keys = list(dct.keys())
    questions = dataset.get_question_prompt(all_keys)
    dct = dataset.interpret_samples(dct)
    return_text = " \n\n## Questions and their correct answers"
    for key in all_keys:
        return_text += "\nQ: " + questions[key] + " A: " + str(dct[key]) + "."
    return return_text
