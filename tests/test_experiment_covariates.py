import pandas as pd
import pytest

from tests.factories import build_experiment, make_arm, make_completed_trial, make_outcome_measure


def make_exp(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50)],
        )
    ]
    return build_experiment(tmp_path, make_completed_trial("NCT1", arms, outcomes))


def test_few_unique_values_mapped_directly(tmp_path):
    exp = make_exp(tmp_path)
    df = pd.DataFrame({
        "Country": ["USA", "UK", "USA"],
        "Country_imputed": ["USA", "UK", "USA"],
        "Duration": ["10", "10", "10"],
        "Duration_imputed": ["10", "10", "10"],
    })
    exp.discretize(df)
    assert exp.options["Country"] == ["USA", "UK"]
    assert df["Country_discretized"].tolist() == [0, 1, 0]


def test_unknown_value_filled_from_imputed_column(tmp_path):
    exp = make_exp(tmp_path)
    df = pd.DataFrame({
        "Country": ["USA", "Unknown"],
        "Country_imputed": ["USA", "UK"],
        "Duration": ["10", "10"],
        "Duration_imputed": ["10", "10"],
    })
    exp.discretize(df)
    assert exp.options["Country"] == ["USA", "UK"]
    assert df["Country_discretized"].tolist() == [0, 1]


def test_many_unique_numeric_splits_on_median(tmp_path):
    exp = make_exp(tmp_path)
    df = pd.DataFrame({
        "Country": ["USA"] * 5,
        "Country_imputed": ["USA"] * 5,
        "Duration": ["10", "20", "30", "40", "Unknown"],
        "Duration_imputed": ["10", "20", "30", "40", "25"],
    })
    exp.discretize(df)
    assert df["Duration_discretized"].tolist() == [0, 0, 1, 1, 0]  # median is 25


def test_many_unique_categorical_buckets_rare_values_as_other(tmp_path):
    exp = make_exp(tmp_path)
    df = pd.DataFrame({
        "Country": ["USA", "USA", "USA", "UK", "France", "Spain"],
        "Country_imputed": ["USA"] * 6,
        "Duration": ["10"] * 6,
        "Duration_imputed": ["10"] * 6,
    })
    exp.discretize(df)
    assert exp.options["Country"] == ["USA", "Other"]
    assert df["Country_discretized"].tolist() == [0, 0, 0, 1, 1, 1]


def test_missing_imputed_column_raises(tmp_path):
    exp = make_exp(tmp_path)
    df = pd.DataFrame({"Country": ["USA"], "Duration": ["10"], "Duration_imputed": ["10"]})
    with pytest.raises(ValueError, match="Country_imputed"):
        exp.discretize(df)


def test_missing_raw_covariate_column_raises(tmp_path):
    exp = make_exp(tmp_path)
    df = pd.DataFrame({"Country_imputed": ["USA"], "Duration": ["10"], "Duration_imputed": ["10"]})
    with pytest.raises(ValueError, match="Country"):
        exp.discretize(df)
