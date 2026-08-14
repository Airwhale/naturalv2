from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import ValidationError

from naturalv2.experiment import Experiment
from naturalv2.pipeline import OUTCOME_COL_NAME, TREATMENT_COL_NAME
from tests.factories import (
    build_experiment,
    make_active_trial,
    make_arm,
    make_completed_trial,
    make_outcome_measure,
)


def _build_continuous_experiment(
    tmp_path,
    *,
    nct_id="NCT012",
    outcome_name="Functional Capacity",
    description=None,
    outcome_bounds=None,
):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcome = make_outcome_measure(outcome_name, "MEAN", "points", [("Drug A", 40, 50)])
    if description is not None:
        outcome["description"] = description
    return build_experiment(
        tmp_path,
        make_completed_trial(nct_id, arms, [outcome]),
        require_binary_endpoint=False,
        outcome_bounds=outcome_bounds,
    )


# -- Arm-type filtering (completed trials) -----------------------------------


def test_placebo_and_sham_arms_excluded(tmp_path):
    arms = [
        make_arm("Drug A", "EXPERIMENTAL"),
        make_arm("Placebo", "PLACEBO_COMPARATOR"),
        make_arm("Sham", "SHAM_COMPARATOR"),
        make_arm("Usual Care", "NO_INTERVENTION"),
        make_arm("Misc", "OTHER"),
    ]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [(a["label"], 10, 50) for a in arms],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT001", arms, outcomes))
    assert exp.treatment_names == ["Drug A"]


def test_combination_arm_with_placebo_in_name_is_kept(tmp_path):
    arms = [
        make_arm("Drug A", "EXPERIMENTAL"),
        make_arm("Drug A + Placebo B", "EXPERIMENTAL"),
        make_arm("Placebo", "PLACEBO_COMPARATOR"),
    ]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [(a["label"], 10, 50) for a in arms],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT002", arms, outcomes))
    assert exp.treatment_names == ["Drug A", "Drug A + Placebo B"]


def test_arm_with_missing_type_excluded(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL"), {"label": "Unknown Arm", "type": None}]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50)],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT003", arms, outcomes))
    assert exp.treatment_names == ["Drug A"]


def test_active_trial_arm_filtering(tmp_path):
    arms = [
        make_arm("Drug A", "EXPERIMENTAL"),
        make_arm("Placebo", "PLACEBO_COMPARATOR"),
    ]
    trial = make_active_trial("NCT004", arms, ["Number of Participants with Response"])
    exp = build_experiment(tmp_path, trial, status="active")
    assert exp.treatment_names == ["Drug A"]
    assert exp.avg_potential_outcomes == []  # no ground truth for active trials


# -- Outcome type / discretization -------------------------------------------


def test_count_outcome_is_binary_and_normalized_by_denom(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50)],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT005", arms, outcomes))
    assert exp.is_binary_outcome(exp.outcome_names[0])
    assert exp.avg_potential_outcomes == [0.2]


def test_percent_unit_normalizes_by_100_regardless_of_param_type(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Mean Percent Change", "MEAN", "percentage", [("Drug A", 40, 50)]
        )
    ]
    exp = build_experiment(
        tmp_path,
        make_completed_trial("NCT006", arms, outcomes),
        require_binary_endpoint=False,
    )
    assert exp.avg_potential_outcomes == [0.4]


def test_continuous_outcome_needs_require_binary_endpoint_false(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Mean Change in Pain Score", "MEAN", "points", [("Drug A", 3.5, 50)]
        )
    ]
    trial = make_completed_trial("NCT007", arms, outcomes)

    # Default require_binary_endpoint=True drops it: title has no binary phrasing.
    exp_default = build_experiment(tmp_path, trial)
    assert exp_default.outcome_names == []

    exp = build_experiment(
        tmp_path,
        make_completed_trial("NCT008", arms, outcomes),
        require_binary_endpoint=False,
    )
    assert exp.outcome_names == ["Mean Change in Pain Score"]
    assert not exp.is_binary_outcome(exp.outcome_names[0])
    assert exp.options[exp.outcome_names[0]] == []
    assert exp.avg_potential_outcomes == [3.5]


def test_active_trial_outcome_falls_back_to_title_heuristic(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    trial = make_active_trial(
        "NCT009", arms, ["Number of Participants with Adverse Events"]
    )
    exp = build_experiment(tmp_path, trial, status="active")
    assert exp.is_binary_outcome(exp.outcome_names[0])


def test_configured_continuous_outcome_range_is_enforced(tmp_path):
    exp = _build_continuous_experiment(
        tmp_path,
        outcome_bounds={"Functional Capacity": {"minimum": 0, "maximum": 55}},
    )
    samples = pd.DataFrame(
        {
            TREATMENT_COL_NAME: ["Drug A"] * 9,
            OUTCOME_COL_NAME: [
                -1,
                0,
                55,
                56,
                4_444_000,
                "not-a-number",
                float("nan"),
                float("inf"),
                -float("inf"),
            ],
        }
    )

    with patch("naturalv2.experiment.logger.warning") as log_warning:
        filtered = exp.discretize_ty(samples, "Functional Capacity")

    assert filtered[OUTCOME_COL_NAME].tolist() == [0, 55]
    assert filtered[f"{OUTCOME_COL_NAME}_discretized"].tolist() == [0, 55]
    assert log_warning.call_args.args[-4:-1] == ("configured", 9, 7)
    assert log_warning.call_args.args[-1] == pytest.approx(7 / 9)
    assert log_warning.call_args.kwargs["extra"] == {
        "phase": "outcome_range_validation",
        "schema_id": "outcome_bounds.v1",
        "status": "rejected",
        "nct_id": "NCT012",
        "outcome": "Functional Capacity",
        "minimum": 0,
        "maximum": 55,
        "bounds_source": "configured",
        "n_sampled": 9,
        "n_rejected": 7,
        "rejection_rate": pytest.approx(7 / 9),
    }


def test_continuous_outcome_range_is_inferred_from_description(tmp_path):
    exp = _build_continuous_experiment(
        tmp_path,
        nct_id="NCT015",
        outcome_name="Fatigue Severity Scale",
        description=(
            "7-item questionnaire assessing fatigue severity. Score range 1-49 "
            "with higher values signifying worse outcome"
        ),
    )

    bounds = exp.outcome_bounds["Fatigue Severity Scale"]
    assert bounds is not None
    assert (bounds.minimum, bounds.maximum, bounds.source) == (1, 49, "description")


def test_configured_outcome_bounds_round_trip_through_yaml(tmp_path):
    exp = _build_continuous_experiment(
        tmp_path,
        nct_id="NCT013",
        outcome_bounds={"Functional Capacity": {"minimum": 0, "maximum": 55}},
    )
    exp._drugbank_names = {"Drug A": []}
    experiment_path = tmp_path / "experiment.yaml"

    exp.to_yaml(str(experiment_path))
    loaded = Experiment.from_yaml(str(experiment_path))

    assert loaded.outcome_bounds["Functional Capacity"] is not None
    assert loaded.outcome_bounds["Functional Capacity"].minimum == 0
    assert loaded.outcome_bounds["Functional Capacity"].maximum == 55
    assert loaded.outcome_bounds["Functional Capacity"].source == "configured"


def test_configured_bounds_reject_unknown_outcome(tmp_path):
    with pytest.raises(ValueError, match="unknown outcome"):
        _build_continuous_experiment(
            tmp_path,
            nct_id="NCT014",
            outcome_bounds={"Typo": {"minimum": 0, "maximum": 55}},
        )


def test_configured_bounds_cannot_override_provenance(tmp_path):
    with pytest.raises(ValidationError, match="source"):
        _build_continuous_experiment(
            tmp_path,
            outcome_bounds={
                "Functional Capacity": {
                    "minimum": 0,
                    "maximum": 55,
                    "source": "description",
                }
            },
        )


def test_null_configured_bounds_disable_inference(tmp_path):
    exp = _build_continuous_experiment(
        tmp_path,
        outcome_name="Fatigue Severity Scale",
        description="The total score ranges from 1 to 49.",
        outcome_bounds={"Fatigue Severity Scale": None},
    )

    assert exp.outcome_bounds["Fatigue Severity Scale"] is None


def test_configured_bounds_reject_binary_outcome(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcome = make_outcome_measure(
        "Number of Participants with Response",
        "COUNT_OF_PARTICIPANTS",
        "Participants",
        [("Drug A", 10, 50)],
    )

    with pytest.raises(ValueError, match="binary outcomes"):
        build_experiment(
            tmp_path,
            make_completed_trial("NCT016", arms, [outcome]),
            outcome_bounds={
                "Number of Participants with Response": {
                    "minimum": 0,
                    "maximum": 1,
                }
            },
        )


# -- APO / ATE ground-truth wiring --------------------------------------------


def test_apo_and_ate_ground_truth_for_two_arms(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL"), make_arm("Drug B", "ACTIVE_COMPARATOR")]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50), ("Drug B", 20, 50)],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT010", arms, outcomes))
    outcome = exp.outcome_names[0]

    assert [outcome, "Drug A"] in exp.apo_outcome_treatment
    assert [outcome, "Drug B"] in exp.apo_outcome_treatment
    assert [outcome, ["Drug A", "Drug B"]] in exp.outcome_treatment

    idx = exp.outcome_treatment.index([outcome, ["Drug A", "Drug B"]])
    assert exp.effect_sizes[idx] == 0.2  # 0.4 - 0.2


def test_single_arm_trial_has_no_ate_ground_truth(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50)],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT011", arms, outcomes))
    assert exp.outcome_treatment == []
    assert exp.apo_outcome_treatment == [[exp.outcome_names[0], "Drug A"]]
