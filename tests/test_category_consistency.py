import pandas as pd

from datasetdna.profiler.category_consistency import (
    check_category_consistency,
)


def test_detects_gender_aliases():
    df = pd.DataFrame({
        "gender": [
            "Male",
            "male",
            "M",
            "Female",
            "female",
            "F",
        ],
    })

    result = check_category_consistency(df)

    assert "gender" in result

    assert result["gender"]["groups"]["male"] == [
        "M",
        "Male",
        "male",
    ]

    assert result["gender"]["groups"]["female"] == [
        "F",
        "Female",
        "female",
    ]

    assert result["gender"]["group_count"] == 2


def test_ignores_consistent_categories():
    df = pd.DataFrame({
        "gender": [
            "Male",
            "Female",
            "Male",
            "Female",
        ],
    })

    result = check_category_consistency(df)

    assert result == {}


def test_ignores_numeric_columns():
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "score": [1, 2, 3, 4],
    })

    result = check_category_consistency(df)

    assert result == {}


def test_detects_boolean_aliases():
    df = pd.DataFrame({
        "active": [
            "Yes",
            "yes",
            "Y",
            "True",
            "No",
            "no",
            "N",
            "False",
        ],
    })

    result = check_category_consistency(df)

    assert "active" in result

    assert result["active"]["groups"]["yes"] == [
        "True",
        "Y",
        "Yes",
        "yes",
    ]

    assert result["active"]["groups"]["no"] == [
        "False",
        "N",
        "No",
        "no",
    ]


def test_ignores_unrelated_categories():
    df = pd.DataFrame({
        "city": [
            "Delhi",
            "Mumbai",
            "Pune",
            "Agra",
        ],
    })

    result = check_category_consistency(df)

    assert result == {}


def test_handles_missing_values():
    df = pd.DataFrame({
        "gender": [
            "Male",
            "male",
            None,
            "M",
        ],
    })

    result = check_category_consistency(df)

    assert "gender" in result
    assert result["gender"]["groups"]["male"] == [
        "M",
        "Male",
        "male",
    ]