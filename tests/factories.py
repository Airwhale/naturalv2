"""Minimal ClinicalTrial JSON builders for Experiment tests."""

import json

from naturalv2.experiment import Experiment


def make_arm(label, arm_type):
    return {"label": label, "type": arm_type}


def make_outcome_measure(title, param_type, unit, groups):
    """``groups``: list of (title, measurement_value, denom_value)."""
    ids = [f"G{i}" for i in range(len(groups))]
    return {
        "type": "PRIMARY",
        "title": title,
        "paramType": param_type,
        "unitOfMeasure": unit,
        "groups": [{"id": gid, "title": g[0]} for gid, g in zip(ids, groups)],
        "denoms": [
            {
                "counts": [
                    {"groupId": gid, "value": str(g[2])} for gid, g in zip(ids, groups)
                ]
            }
        ],
        "classes": [
            {
                "categories": [
                    {
                        "measurements": [
                            {"groupId": gid, "value": str(g[1])}
                            for gid, g in zip(ids, groups)
                        ]
                    }
                ]
            }
        ],
    }


def make_completed_trial(nct_id, arms, outcome_measures):
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": "t",
                "organization": {"fullName": "org"},
            },
            "statusModule": {
                "statusVerifiedDate": "2024-01",
                "overallStatus": "COMPLETED",
            },
            "armsInterventionsModule": {"armGroups": arms},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": o["title"]} for o in outcome_measures]
            },
        },
        "resultsSection": {
            "participantFlowModule": {},
            "baselineCharacteristicsModule": {"groups": [], "measures": []},
            "outcomeMeasuresModule": {"outcomeMeasures": outcome_measures},
        },
        "derivedSection": {},
        "hasResults": True,
    }


def make_active_trial(nct_id, arms, outcome_titles):
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": "t",
                "organization": {"fullName": "org"},
            },
            "statusModule": {
                "statusVerifiedDate": "2024-01",
                "overallStatus": "RECRUITING",
            },
            "armsInterventionsModule": {"armGroups": arms},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": t} for t in outcome_titles]
            },
        },
        "derivedSection": {},
        "hasResults": False,
    }


def build_experiment(tmp_path, trial_json, status="completed", **experiment_kwargs):
    """Write ``trial_json`` to disk and load it as an Experiment."""
    nct_id = trial_json["protocolSection"]["identificationModule"]["nctId"]
    subdir = "nct_reports_test" if status == "active" else "nct_reports"
    trial_dir = tmp_path / subdir
    trial_dir.mkdir(exist_ok=True)
    (trial_dir / f"{nct_id}.json").write_text(json.dumps(trial_json))
    return Experiment(
        data_path=str(tmp_path),
        nct_id=nct_id,
        experiment_name="exp",
        status=status,
        **experiment_kwargs,
    )
