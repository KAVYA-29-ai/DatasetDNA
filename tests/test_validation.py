import pandas as pd
import pytest

from datasetdna.profiler.types import check_types

from datasetdna.scoring.health_score import (
    calculate_data_quality_score,
    _get_quality_findings,
)

from datasetdna.utils.helpers import (
    clean_numeric_like_columns,
    normalize_missing_values,
    validate_dataframe,
)


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def test_none_dataframe_is_rejected():
    with pytest.raises(
        ValueError,
        match="DataFrame cannot be None",
    ):
        validate_dataframe(None)


def test_empty_dataframe_is_rejected():
    df = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="CSV contains no data rows",
    ):
        validate_dataframe(df)


def test_dataframe_without_columns_is_rejected():
    df = pd.DataFrame(index=[0, 1, 2])

    with pytest.raises(
        ValueError,
        match="CSV contains no columns",
    ):
        validate_dataframe(df)


def test_valid_dataframe_is_accepted():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
            "salary": [30000, 40000, 50000],
        }
    )

    validate_dataframe(df)


def test_single_row_dataframe_is_valid():
    df = pd.DataFrame(
        {
            "age": [20],
            "salary": [30000],
        }
    )

    validate_dataframe(df)


def test_single_column_dataframe_is_valid():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
        }
    )

    validate_dataframe(df)


def test_all_null_dataframe_is_structurally_valid():
    df = pd.DataFrame(
        {
            "age": [None, None, None],
            "city": [None, None, None],
        }
    )

    validate_dataframe(df)


# ============================================================
# MISSING VALUE NORMALIZATION
# ============================================================

def test_common_missing_values_are_normalized():
    df = pd.DataFrame(
        {
            "value": [
                "",
                " ",
                "N/A",
                "NULL",
                "unknown",
                "?",
                "-",
                "valid",
            ]
        }
    )

    result = normalize_missing_values(df)

    assert result["value"].isna().sum() == 7
    assert result["value"].iloc[7] == "valid"


def test_string_whitespace_is_stripped():
    df = pd.DataFrame(
        {
            "city": [
                " Delhi ",
                " Mumbai",
                "Bangalore ",
            ]
        }
    )

    result = normalize_missing_values(df)

    assert result["city"].tolist() == [
        "Delhi",
        "Mumbai",
        "Bangalore",
    ]


# ============================================================
# NUMERIC-LIKE VALUE CLEANING
# ============================================================

def test_currency_values_are_converted():
    df = pd.DataFrame(
        {
            "salary": [
                "₹50,000",
                "₹60,000",
                "₹70,000",
            ]
        }
    )

    result = clean_numeric_like_columns(df)

    assert pd.api.types.is_numeric_dtype(
        result["salary"]
    )

    assert result["salary"].tolist() == [
        50000.0,
        60000.0,
        70000.0,
    ]


def test_dollar_values_are_converted():
    df = pd.DataFrame(
        {
            "price": [
                "$1,000",
                "$2,500",
                "$5,000",
            ]
        }
    )

    result = clean_numeric_like_columns(df)

    assert pd.api.types.is_numeric_dtype(
        result["price"]
    )

    assert result["price"].tolist() == [
        1000.0,
        2500.0,
        5000.0,
    ]


def test_percentage_values_are_converted():
    df = pd.DataFrame(
        {
            "conversion_rate": [
                "10%",
                "25%",
                "50%",
            ]
        }
    )

    result = clean_numeric_like_columns(df)

    assert pd.api.types.is_numeric_dtype(
        result["conversion_rate"]
    )

    assert result["conversion_rate"].tolist() == [
        10.0,
        25.0,
        50.0,
    ]


def test_comma_separated_numbers_are_converted():
    df = pd.DataFrame(
        {
            "population": [
                "1,000",
                "25,000",
                "100,000",
            ]
        }
    )

    result = clean_numeric_like_columns(df)

    assert pd.api.types.is_numeric_dtype(
        result["population"]
    )

    assert result["population"].tolist() == [
        1000.0,
        25000.0,
        100000.0,
    ]


def test_numeric_like_column_with_some_invalid_values_is_cleaned():
    df = pd.DataFrame(
        {
            "salary": [
                "50000",
                "60000",
                "invalid",
                "70000",
                "80000",
            ]
        }
    )

    result = clean_numeric_like_columns(df)

    assert pd.api.types.is_numeric_dtype(
        result["salary"]
    )

    assert result["salary"].iloc[0] == 50000.0
    assert result["salary"].iloc[1] == 60000.0
    assert pd.isna(result["salary"].iloc[2])
    assert result["salary"].iloc[3] == 70000.0
    assert result["salary"].iloc[4] == 80000.0


def test_mostly_categorical_column_is_not_forced_to_numeric():
    df = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Mumbai",
                "Bangalore",
                "Chennai",
                "Pune",
            ]
        }
    )

    result = clean_numeric_like_columns(df)

    assert not pd.api.types.is_numeric_dtype(
        result["city"]
    )


# ============================================================
# MIXED DATA
# ============================================================

def test_mixed_numeric_column_is_cleaned():
    df = pd.DataFrame(
        {
            "amount": [
                "1000",
                "2000",
                "3000",
                "4000",
                "not_available",
            ]
        }
    )

    result = clean_numeric_like_columns(df)

    assert pd.api.types.is_numeric_dtype(
        result["amount"]
    )

    assert result["amount"].iloc[0] == 1000.0
    assert result["amount"].iloc[1] == 2000.0
    assert result["amount"].iloc[2] == 3000.0
    assert result["amount"].iloc[3] == 4000.0
    assert pd.isna(result["amount"].iloc[4])


# ============================================================
# VALIDATION EDGE CASES
# ============================================================

def test_constant_dataframe_is_structurally_valid():
    df = pd.DataFrame(
        {
            "age": [25, 25, 25, 25],
            "city": [
                "Delhi",
                "Delhi",
                "Delhi",
                "Delhi",
            ],
        }
    )

    validate_dataframe(df)


def test_duplicate_heavy_dataframe_is_structurally_valid():
    df = pd.DataFrame(
        {
            "age": [20, 20, 20, 21],
            "city": [
                "Delhi",
                "Delhi",
                "Delhi",
                "Mumbai",
            ],
        }
    )

    validate_dataframe(df)


def test_single_row_with_missing_values_is_valid():
    df = pd.DataFrame(
        {
            "age": [None],
            "city": [None],
            "salary": [None],
        }
    )

    validate_dataframe(df)


def test_numeric_dataframe_is_not_modified_unnecessarily():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
            "salary": [30000, 40000, 50000],
        }
    )

    result = clean_numeric_like_columns(df)

    assert result["age"].tolist() == [
        20,
        21,
        22,
    ]

    assert result["salary"].tolist() == [
        30000,
        40000,
        50000,
    ]


# ============================================================
# BOOLEAN TYPE DETECTION
# ============================================================

def test_boolean_values_are_detected():
    df = pd.DataFrame(
        {
            "active": [
                "yes",
                "no",
                "yes",
                "no",
            ]
        }
    )

    result = check_types(df)

    assert result["active"]["detected_type"] == "boolean"
    assert result["active"]["confidence"] == 1.0


def test_boolean_values_are_case_insensitive():
    df = pd.DataFrame(
        {
            "active": [
                "YES",
                "No",
                "TRUE",
                "false",
            ]
        }
    )

    result = check_types(df)

    assert result["active"]["detected_type"] == "boolean"


def test_short_boolean_values_are_detected():
    df = pd.DataFrame(
        {
            "active": [
                "Y",
                "N",
                "Y",
                "N",
            ]
        }
    )

    result = check_types(df)

    assert result["active"]["detected_type"] == "boolean"


def test_true_false_values_are_detected():
    df = pd.DataFrame(
        {
            "verified": [
                "true",
                "false",
                "true",
                "false",
            ]
        }
    )

    result = check_types(df)

    assert result["verified"]["detected_type"] == "boolean"


def test_boolean_values_with_missing_values_are_detected():
    df = pd.DataFrame(
        {
            "active": [
                "yes",
                "no",
                None,
                "yes",
                "no",
            ]
        }
    )

    result = check_types(df)

    assert result["active"]["detected_type"] == "boolean"
    assert result["active"]["confidence"] == 1.0


# ============================================================
# DATE TYPE DETECTION
# ============================================================

def test_iso_dates_are_detected():
    df = pd.DataFrame(
        {
            "signup_date": [
                "2026-01-10",
                "2026-02-15",
                "2026-03-20",
                "2026-04-25",
            ]
        }
    )

    result = check_types(df)

    assert result["signup_date"]["detected_type"] == "date"
    assert result["signup_date"]["confidence"] == 1.0


def test_slash_dates_are_detected():
    df = pd.DataFrame(
        {
            "signup_date": [
                "10/01/2026",
                "15/02/2026",
                "20/03/2026",
                "25/04/2026",
            ]
        }
    )

    result = check_types(df)

    assert result["signup_date"]["detected_type"] == "date"


def test_text_dates_are_detected():
    df = pd.DataFrame(
        {
            "signup_date": [
                "Jan 10 2026",
                "Feb 15 2026",
                "Mar 20 2026",
                "Apr 25 2026",
            ]
        }
    )

    result = check_types(df)

    assert result["signup_date"]["detected_type"] == "date"


def test_dates_with_missing_values_are_detected():
    df = pd.DataFrame(
        {
            "signup_date": [
                "2026-01-10",
                None,
                "2026-03-20",
                "2026-04-25",
            ]
        }
    )

    result = check_types(df)

    assert result["signup_date"]["detected_type"] == "date"


def test_invalid_dates_do_not_become_date_type():
    df = pd.DataFrame(
        {
            "signup_date": [
                "2026-01-10",
                "not-a-date",
                "hello",
                "unknown",
            ]
        }
    )

    result = check_types(df)

    assert result["signup_date"]["detected_type"] != "date"


# ============================================================
# SEMANTIC INVALID VALUE DETECTION
# ============================================================

def _base_quality_results(numerical):
    return {
        "missing": {
            "total_missing_cells": 0,
            "columns": {},
        },
        "duplicates": {
            "duplicate_count": 0,
            "duplicate_percentage": 0.0,
        },
        "target": {
            "provided": False,
            "column": None,
        },
        "cardinality": {},
        "numerical": numerical,
    }


def _has_invalid_value_finding(findings):
    return any(
        finding.get("type") == "invalid_values"
        for finding in findings
    )


def test_invalid_age_above_120_is_detected():
    results = _base_quality_results(
        {
            "age": {
                "count": 4,
                "min": 20.0,
                "max": 150.0,
            }
        }
    )

    findings = _get_quality_findings(results)

    assert _has_invalid_value_finding(findings)


def test_invalid_negative_age_is_detected():
    results = _base_quality_results(
        {
            "age": {
                "count": 4,
                "min": -5.0,
                "max": 40.0,
            }
        }
    )

    findings = _get_quality_findings(results)

    assert _has_invalid_value_finding(findings)


def test_percentage_above_100_is_detected():
    results = _base_quality_results(
        {
            "conversion_percentage": {
                "count": 4,
                "min": 10.0,
                "max": 150.0,
            }
        }
    )

    findings = _get_quality_findings(results)

    assert _has_invalid_value_finding(findings)


def test_probability_above_1_is_detected():
    results = _base_quality_results(
        {
            "probability": {
                "count": 4,
                "min": 0.1,
                "max": 2.0,
            }
        }
    )

    findings = _get_quality_findings(results)

    assert _has_invalid_value_finding(findings)


def test_negative_salary_is_detected():
    results = _base_quality_results(
        {
            "salary": {
                "count": 4,
                "min": -500.0,
                "max": 50000.0,
            }
        }
    )

    findings = _get_quality_findings(results)

    assert _has_invalid_value_finding(findings)


def test_valid_age_range_is_not_flagged():
    results = _base_quality_results(
        {
            "age": {
                "count": 4,
                "min": 18.0,
                "max": 80.0,
            }
        }
    )

    findings = _get_quality_findings(results)

    invalid_findings = [
        finding
        for finding in findings
        if finding.get("type") == "invalid_values"
    ]

    assert invalid_findings == []


def test_valid_percentage_range_is_not_flagged():
    results = _base_quality_results(
        {
            "conversion_percentage": {
                "count": 4,
                "min": 10.0,
                "max": 95.0,
            }
        }
    )

    findings = _get_quality_findings(results)

    invalid_findings = [
        finding
        for finding in findings
        if finding.get("type") == "invalid_values"
    ]

    assert invalid_findings == []


def test_valid_probability_range_is_not_flagged():
    results = _base_quality_results(
        {
            "probability": {
                "count": 4,
                "min": 0.1,
                "max": 0.9,
            }
        }
    )

    findings = _get_quality_findings(results)

    invalid_findings = [
        finding
        for finding in findings
        if finding.get("type") == "invalid_values"
    ]

    assert invalid_findings == []


# ============================================================
# DATA QUALITY SCORE — INVALID VALUES
# ============================================================

def test_invalid_values_reduce_data_quality_score():
    results = _base_quality_results(
        {
            "age": {
                "count": 4,
                "min": -5.0,
                "max": 150.0,
            }
        }
    )

    score = calculate_data_quality_score(results)

    assert score < 100

def test_time_like_text_does_not_trigger_date_parsing_warning():
    import warnings

    df = pd.DataFrame({
        "time_label": [
            "3 A.M.",
            "5 A.M.",
            "7 P.M.",
            "10 P.M.",
        ]
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        result = check_types(df)

    assert result["time_label"]["detected_type"] == "categorical"

    future_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning)
    ]

    assert not future_warnings