import pandas as pd

from datasetdna.engine.profiler import profile_dataframe
from datasetdna.recommendations.recommendations import (
    generate_recommendations,
)


def test_missing_value_recommendation():
    results = {
        "missing": {
            "columns": {
                "age": {
                    "percentage": 10,
                }
            }
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "missing"
    assert recommendations[0]["severity"] == "medium"
    assert "age" in recommendations[0]["message"]


def test_high_missing_value_recommendation():
    results = {
        "missing": {
            "columns": {
                "salary": {
                    "percentage": 25,
                }
            }
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["severity"] == "high"
    assert "salary" in recommendations[0]["message"]


def test_duplicate_recommendation():
    results = {
        "duplicates": {
            "duplicate_percentage": 5,
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "duplicates"
    assert recommendations[0]["severity"] == "medium"


def test_target_imbalance_recommendation():
    results = {
        "target": {
            "provided": True,
            "column": "is_active",
            "imbalance_ratio": 3,
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "target_imbalance"
    assert recommendations[0]["severity"] == "medium"
    assert "is_active" in recommendations[0]["message"]


def test_invalid_value_recommendation():
    results = {
        "health": {
            "quality_findings": [
                {
                    "type": "invalid_values",
                    "severity": "high",
                    "column": "age",
                    "value": "[-5, 150]",
                }
            ]
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "invalid_values"
    assert recommendations[0]["severity"] == "high"
    assert "age" in recommendations[0]["message"]


def test_constant_column_recommendation():
    results = {
        "health": {
            "quality_findings": [
                {
                    "type": "constant",
                    "severity": "medium",
                    "column": "constant_column",
                    "value": 1,
                }
            ]
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "constant"
    assert "constant_column" in recommendations[0]["message"]


def test_strong_correlation_recommendation():
    results = {
        "health": {
            "statistical_signals": [
                {
                    "type": "correlation",
                    "severity": "high",
                    "columns": ["age", "salary"],
                    "value": 0.949,
                }
            ]
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "correlation"
    assert recommendations[0]["severity"] == "high"
    assert "age" in recommendations[0]["message"]
    assert "salary" in recommendations[0]["message"]


def test_outlier_recommendation():
    results = {
        "health": {
            "statistical_signals": [
                {
                    "type": "outliers",
                    "severity": "high",
                    "column": "age",
                    "value": 11.11,
                }
            ]
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "outliers"
    assert recommendations[0]["severity"] == "high"


def test_skewness_recommendation():
    results = {
        "health": {
            "statistical_signals": [
                {
                    "type": "skewness",
                    "severity": "high",
                    "column": "salary",
                    "value": 4.11,
                }
            ]
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "skewness"
    assert recommendations[0]["severity"] == "high"


def test_cardinality_recommendation():
    results = {
        "health": {
            "statistical_signals": [
                {
                    "type": "cardinality",
                    "severity": "medium",
                    "column": "name",
                    "value": 95,
                }
            ]
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1
    assert recommendations[0]["type"] == "cardinality"
    assert recommendations[0]["severity"] == "medium"


def test_no_recommendations_for_clean_dataset():
    results = {
        "missing": {
            "columns": {
                "age": {
                    "percentage": 0,
                }
            }
        },
        "duplicates": {
            "duplicate_percentage": 0,
        },
        "target": {
            "provided": True,
            "column": "species",
            "imbalance_ratio": 1,
        },
        "health": {
            "quality_findings": [],
            "statistical_signals": [],
        },
    }

    recommendations = generate_recommendations(results)

    assert recommendations == []


def test_multiple_recommendations():
    results = {
        "missing": {
            "columns": {
                "age": {
                    "percentage": 10,
                }
            }
        },
        "duplicates": {
            "duplicate_percentage": 5,
        },
        "target": {
            "provided": True,
            "column": "target",
            "imbalance_ratio": 3,
        },
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 3
def test_mixed_type_recommendation():
    results = {
        "mixed_types": {
            "age": {
                "types": {
                    "int": 3,
                    "str": 1,
                },
                "type_count": 2,
            }
        }
    }

    recommendations = generate_recommendations(results)

    assert len(recommendations) == 1

    recommendation = recommendations[0]

    assert recommendation["type"] == "mixed_types"
    assert recommendation["severity"] == "medium"
    assert recommendation["column"] == "age"
    assert recommendation["value"] == "int, str"
    assert "Standardize the values" in recommendation["message"]

def test_category_consistency_recommendation():
    df = pd.DataFrame(
        {
            "gender": [
                "Male",
                "male",
                "M",
                "Female",
                "female",
                "F",
            ]
        }
    )

    results = profile_dataframe(df)

    recommendations = results["recommendations"]

    category_recommendations = [
        recommendation
        for recommendation in recommendations
        if recommendation["type"] == "category_consistency"
    ]

    assert len(category_recommendations) == 1
    assert category_recommendations[0]["column"] == "gender"
    assert category_recommendations[0]["severity"] == "medium"
    assert "Male" in category_recommendations[0]["message"]
    assert "male" in category_recommendations[0]["message"]
    assert "M" in category_recommendations[0]["message"]