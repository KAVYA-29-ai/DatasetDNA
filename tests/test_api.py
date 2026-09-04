import pandas as pd

import datasetdna as dna


def test_profile_api():
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "income": [100, 200, 300, 400],
        "churn": [0, 0, 1, 1],
    })

    result = dna.profile(
        df,
        target="churn",
    )

    assert isinstance(result, dict)
    assert "overview" in result
    assert "schema" in result
    assert "missing" in result
    assert "duplicates" in result
    assert "cardinality" in result
    assert "numerical" in result
    assert "categorical" in result
    assert "outliers" in result
    assert "correlations" in result
    assert "target" in result
    assert "health" in result
    assert "recommendations" in result

    assert result["target"]["task_type"] == "classification"


def test_profile_api_auto_infers_target():
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "income": [100, 200, 300, 400],
        "churn": [0, 0, 1, 1],
    })

    result = dna.profile(df)

    assert result["target"]["column"] == "churn"
    assert result["target"]["inferred"] is True


def test_profile_api_regression():
    df = pd.DataFrame({
        "age": list(range(25)),
        "score": [float(i) for i in range(25)],
    })

    result = dna.profile(
        df,
        target="score",
    )

    assert result["target"]["task_type"] == "regression"
    assert result["target"]["class_count"] is None
    assert result["target"]["imbalance_ratio"] is None