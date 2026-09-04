import pandas as pd

from datasetdna.profiler.mixed_types import check_mixed_types


def test_detects_mixed_types():
    df = pd.DataFrame({
        "age": [21, 22, "unknown", 25],
    })

    result = check_mixed_types(df)

    assert "age" in result
    assert result["age"]["type_count"] == 2
    assert result["age"]["types"]["int"] == 3
    assert result["age"]["types"]["str"] == 1


def test_ignores_consistent_columns():
    df = pd.DataFrame({
        "age": [21, 22, 25, 30],
        "name": ["A", "B", "C", "D"],
    })

    result = check_mixed_types(df)

    assert result == {}


def test_ignores_null_values():
    df = pd.DataFrame({
        "age": [21, 22, None, 25],
    })

    result = check_mixed_types(df)

    assert result == {}


def test_handles_multiple_mixed_columns():
    df = pd.DataFrame({
        "age": [21, "unknown", 25],
        "salary": [50000, "N/A", 70000],
    })

    result = check_mixed_types(df)

    assert "age" in result
    assert "salary" in result


def test_handles_all_null_column():
    df = pd.DataFrame({
        "age": [None, None, None],
    })

    result = check_mixed_types(df)

    assert result == {}