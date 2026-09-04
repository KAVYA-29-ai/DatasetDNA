import pandas as pd
import pytest

from datasetdna.profiler.overview import check_overview
from datasetdna.profiler.schema import check_schema
from datasetdna.profiler.missing import check_missing
from datasetdna.profiler.duplicates import check_duplicates
from datasetdna.profiler.cardinality import check_cardinality
from datasetdna.profiler.numerical import check_numerical
from datasetdna.profiler.categorical import check_categorical
from datasetdna.profiler.outliers import check_outliers
from datasetdna.profiler.correlations import check_correlations
from datasetdna.profiler.target import check_target


# ============================================================
# SHARED DATA
# ============================================================

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
            "salary": [30000, 40000, 50000, 60000, 70000],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
            ],
            "active": [
                True,
                False,
                True,
                True,
                False,
            ],
        }
    )


# ============================================================
# OVERVIEW
# ============================================================

def test_overview_returns_row_count(sample_dataframe):
    result = check_overview(sample_dataframe)

    assert result["rows"] == 5


def test_overview_returns_column_count(sample_dataframe):
    result = check_overview(sample_dataframe)

    assert result["columns"] == 4


def test_overview_returns_memory_usage(sample_dataframe):
    result = check_overview(sample_dataframe)

    assert "memory_usage_mb" in result
    assert result["memory_usage_mb"] >= 0


def test_overview_empty_dataframe():
    df = pd.DataFrame()

    result = check_overview(df)

    assert result["rows"] == 0
    assert result["columns"] == 0
    assert result["memory_usage_mb"] >= 0


# ============================================================
# SCHEMA
# ============================================================

def test_schema_returns_all_columns(sample_dataframe):
    result = check_schema(sample_dataframe)

    assert set(result.keys()) == {
        "age",
        "salary",
        "city",
        "active",
    }


def test_schema_detects_numeric_columns(sample_dataframe):
    result = check_schema(sample_dataframe)

    assert result["age"] == "int64"
    assert result["salary"] == "int64"


def test_schema_detects_string_column(sample_dataframe):
    result = check_schema(sample_dataframe)

    assert result["city"] == "str"


def test_schema_detects_boolean_column(sample_dataframe):
    result = check_schema(sample_dataframe)

    assert result["active"] == "bool"


# ============================================================
# MISSING VALUES
# ============================================================

def test_missing_detects_missing_cells():
    df = pd.DataFrame(
        {
            "age": [20, None, 22],
            "city": ["Delhi", "Mumbai", None],
        }
    )

    result = check_missing(df)

    assert result["total_missing_cells"] == 2


def test_missing_returns_column_details():
    df = pd.DataFrame(
        {
            "age": [20, None, 22, None],
            "city": ["Delhi", "Mumbai", "Pune", "Delhi"],
        }
    )

    result = check_missing(df)

    assert result["columns"]["age"]["count"] == 2
    assert result["columns"]["age"]["percentage"] == 50.0

    assert result["columns"]["city"]["count"] == 0
    assert result["columns"]["city"]["percentage"] == 0.0


def test_missing_handles_no_missing_values(sample_dataframe):
    result = check_missing(sample_dataframe)

    assert result["total_missing_cells"] == 0

    for column in sample_dataframe.columns:
        assert result["columns"][column]["count"] == 0
        assert result["columns"][column]["percentage"] == 0.0


def test_missing_handles_all_null_column():
    df = pd.DataFrame(
        {
            "age": [None, None, None],
            "city": ["Delhi", "Mumbai", "Pune"],
        }
    )

    result = check_missing(df)

    assert result["columns"]["age"]["count"] == 3
    assert result["columns"]["age"]["percentage"] == 100.0


# ============================================================
# DUPLICATES
# ============================================================

def test_duplicates_detects_duplicate_rows():
    df = pd.DataFrame(
        {
            "age": [20, 21, 20, 22],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
            ],
        }
    )

    result = check_duplicates(df)

    assert result["duplicate_count"] == 1
    assert result["duplicate_percentage"] == 25.0


def test_duplicates_returns_zero_when_no_duplicates():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
            "city": [
                "Delhi",
                "Mumbai",
                "Pune",
            ],
        }
    )

    result = check_duplicates(df)

    assert result["duplicate_count"] == 0
    assert result["duplicate_percentage"] == 0.0


def test_duplicates_ignores_identifier_column():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [20, 20, 20],
            "city": [
                "Delhi",
                "Delhi",
                "Delhi",
            ],
        }
    )

    result = check_duplicates(df)

    assert result["duplicate_count"] == 2
    assert "id" in result["ignored_identifier_columns"]


def test_duplicates_ignores_suffix_id_column():
    df = pd.DataFrame(
        {
            "customer_id": [101, 102, 103],
            "age": [20, 20, 20],
        }
    )

    result = check_duplicates(df)

    assert "customer_id" in result["ignored_identifier_columns"]
    assert result["duplicate_count"] == 2


def test_duplicates_handles_empty_dataframe():
    df = pd.DataFrame()

    result = check_duplicates(df)

    assert result["duplicate_count"] == 0
    assert result["duplicate_percentage"] == 0.0


# ============================================================
# CARDINALITY
# ============================================================

def test_cardinality_counts_unique_values(sample_dataframe):
    result = check_cardinality(sample_dataframe)

    assert result["age"]["unique_count"] == 5
    assert result["salary"]["unique_count"] == 5
    assert result["city"]["unique_count"] == 3


def test_cardinality_calculates_unique_percentage(
    sample_dataframe,
):
    result = check_cardinality(sample_dataframe)

    assert result["age"]["unique_percentage"] == 100.0
    assert result["city"]["unique_percentage"] == 60.0


def test_cardinality_detects_semantic_type(sample_dataframe):
    result = check_cardinality(sample_dataframe)

    assert result["age"]["detected_type"] == "numeric"
    assert result["city"]["detected_type"] == "categorical"


def test_cardinality_detects_id_like_column():
    df = pd.DataFrame(
        {
            "customer_id": [
                1001,
                1002,
                1003,
                1004,
            ],
            "age": [20, 21, 22, 23],
        }
    )

    result = check_cardinality(df)

    assert result["customer_id"]["is_id_like"] is True
    assert result["customer_id"]["detected_type"] == "id"


# ============================================================
# NUMERICAL
# ============================================================

def test_numerical_analyzes_numeric_columns(
    sample_dataframe,
):
    result = check_numerical(sample_dataframe)

    assert "age" in result
    assert "salary" in result


def test_numerical_ignores_categorical_columns(
    sample_dataframe,
):
    result = check_numerical(sample_dataframe)

    assert "city" not in result


def test_numerical_calculates_mean():
    df = pd.DataFrame(
        {
            "age": [10, 20, 30, 40, 50],
        }
    )

    result = check_numerical(df)

    assert result["age"]["mean"] == 30.0


def test_numerical_calculates_median():
    df = pd.DataFrame(
        {
            "age": [10, 20, 30, 40, 50],
        }
    )

    result = check_numerical(df)

    assert result["age"]["median"] == 30.0


def test_numerical_calculates_min_max():
    df = pd.DataFrame(
        {
            "age": [10, 20, 30, 40, 50],
        }
    )

    result = check_numerical(df)

    assert result["age"]["min"] == 10.0
    assert result["age"]["max"] == 50.0


def test_numerical_calculates_count_with_missing_values():
    df = pd.DataFrame(
        {
            "age": [10, None, 30, None, 50],
        }
    )

    result = check_numerical(df)

    assert result["age"]["count"] == 3


def test_numerical_ignores_all_null_column():
    df = pd.DataFrame(
        {
            "age": [None, None, None],
        }
    )

    result = check_numerical(df)

    assert "age" not in result


# ============================================================
# CATEGORICAL
# ============================================================

def test_categorical_analyzes_categorical_columns(
    sample_dataframe,
):
    result = check_categorical(sample_dataframe)

    assert "city" in result


def test_categorical_ignores_numeric_columns(
    sample_dataframe,
):
    result = check_categorical(sample_dataframe)

    assert "age" not in result
    assert "salary" not in result


def test_categorical_counts_categories():
    df = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Delhi",
                "Mumbai",
                "Pune",
                "Delhi",
            ]
        }
    )

    result = check_categorical(df)

    assert result["city"]["unique_count"] == 3
    assert result["city"]["total_values"] == 5

    assert result["city"]["categories"]["Delhi"]["count"] == 3
    assert result["city"]["categories"]["Mumbai"]["count"] == 1
    assert result["city"]["categories"]["Pune"]["count"] == 1


def test_categorical_calculates_percentages():
    df = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Delhi",
                "Mumbai",
                "Pune",
            ]
        }
    )

    result = check_categorical(df)

    assert (
        result["city"]["categories"]["Delhi"]["percentage"]
        == 50.0
    )


def test_categorical_handles_missing_values():
    df = pd.DataFrame(
        {
            "city": [
                "Delhi",
                None,
                "Mumbai",
                None,
            ]
        }
    )

    result = check_categorical(df)

    assert result["city"]["total_values"] == 2
    assert result["city"]["unique_count"] == 2


def test_categorical_ignores_all_null_column():
    df = pd.DataFrame(
        {
            "city": [
                None,
                None,
                None,
            ]
        }
    )

    result = check_categorical(df)

    assert "city" not in result


# ============================================================
# OUTLIERS
# ============================================================

def test_outliers_detects_extreme_value():
    df = pd.DataFrame(
        {
            "salary": [
                30000,
                32000,
                31000,
                30500,
                1000000,
            ]
        }
    )

    result = check_outliers(df)

    assert result["salary"]["outlier_count"] >= 1


def test_outliers_returns_percentage():
    df = pd.DataFrame(
        {
            "salary": [
                30000,
                32000,
                31000,
                30500,
                1000000,
            ]
        }
    )

    result = check_outliers(df)

    assert "outlier_percentage" in result["salary"]
    assert result["salary"]["outlier_percentage"] >= 0


def test_outliers_returns_iqr_statistics():
    df = pd.DataFrame(
        {
            "salary": [
                10,
                20,
                30,
                40,
                50,
            ]
        }
    )

    result = check_outliers(df)

    assert "q1" in result["salary"]
    assert "q3" in result["salary"]
    assert "iqr" in result["salary"]
    assert "lower_bound" in result["salary"]
    assert "upper_bound" in result["salary"]


def test_outliers_ignores_categorical_columns():
    df = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Mumbai",
                "Pune",
                "Delhi",
            ]
        }
    )

    result = check_outliers(df)

    assert "city" not in result


def test_outliers_handles_missing_values():
    df = pd.DataFrame(
        {
            "salary": [
                30000,
                None,
                40000,
                50000,
                None,
            ]
        }
    )

    result = check_outliers(df)

    assert result["salary"]["outlier_count"] == 0


# ============================================================
# CORRELATIONS
# ============================================================

def test_correlations_returns_expected_structure(
    sample_dataframe,
):
    result = check_correlations(sample_dataframe)

    assert "numeric" in result
    assert "categorical" in result


def test_correlations_detects_numeric_relationship():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        }
    )

    result = check_correlations(df)

    assert len(result["numeric"]) == 1

    correlation = result["numeric"][0]

    assert correlation["column_1"] == "x"
    assert correlation["column_2"] == "y"
    assert correlation["pearson"] == 1.0


def test_correlations_detects_negative_relationship():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 8, 6, 4, 2],
        }
    )

    result = check_correlations(df)

    assert len(result["numeric"]) == 1
    assert result["numeric"][0]["pearson"] == -1.0


def test_correlations_skips_constant_numeric_columns():
    df = pd.DataFrame(
        {
            "x": [1, 1, 1, 1],
            "y": [1, 2, 3, 4],
        }
    )

    result = check_correlations(df)

    assert result["numeric"] == []


def test_correlations_detects_categorical_relationship():
    df = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Delhi",
                "Mumbai",
                "Mumbai",
                "Pune",
                "Pune",
            ],
            "segment": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
        }
    )

    result = check_correlations(df)

    assert len(result["categorical"]) == 1

    correlation = result["categorical"][0]

    assert correlation["column_1"] == "city"
    assert correlation["column_2"] == "segment"
    assert 0.0 <= correlation["cramers_v"] <= 1.0


def test_correlations_skips_high_cardinality_categorical_columns():
    df = pd.DataFrame(
        {
            "email": [
                "a@test.com",
                "b@test.com",
                "c@test.com",
                "d@test.com",
                "e@test.com",
            ],
            "city": [
                "Delhi",
                "Mumbai",
                "Pune",
                "Delhi",
                "Mumbai",
            ],
        }
    )

    result = check_correlations(df)

    assert result["categorical"] == []


# ============================================================
# TARGET
# ============================================================

def test_target_without_target_column():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
            "salary": [30000, 40000, 50000],
        }
    )

    result = check_target(df, None)

    assert result["provided"] is False
    assert result["column"] is None


def test_target_detects_numeric_target():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
            "salary": [30000, 40000, 50000],
            "target": [1, 0, 1],
        }
    )

    result = check_target(df, "target")

    assert result["provided"] is True
    assert result["column"] == "target"


def test_target_detects_categorical_target():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "target": [
                "yes",
                "no",
                "yes",
                "no",
            ],
        }
    )

    result = check_target(df, "target")

    assert result["provided"] is True
    assert result["column"] == "target"


def test_target_calculates_class_distribution():
    df = pd.DataFrame(
        {
            "target": [
                "yes",
                "yes",
                "yes",
                "no",
            ]
        }
    )

    result = check_target(df, "target")

    assert "class_distribution" in result

    assert (
        result["class_distribution"]["yes"]["count"]
        == 3
    )

    assert (
        result["class_distribution"]["no"]["count"]
        == 1
    )

    assert (
        result["class_distribution"]["yes"]["percentage"]
        == 75.0
    )

    assert (
        result["class_distribution"]["no"]["percentage"]
        == 25.0
    )


def test_target_detects_missing_target_values():
    df = pd.DataFrame(
        {
            "target": [
                "yes",
                "no",
                None,
                "yes",
            ]
        }
    )

    result = check_target(df, "target")

    assert result["provided"] is True
    assert result["column"] == "target"
    assert result["class_count"] == 2


def test_target_with_nonexistent_column_returns_target_metadata():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
        }
    )

    result = check_target(df, "target")

    assert result["provided"] is True
    assert result["column"] == "target"


# ============================================================
# TARGET TASK DETECTION
# ============================================================

def test_numeric_target_with_many_unique_values_is_regression():
    df = pd.DataFrame({
        "track_popularity": list(range(100))
    })

    result = check_target(
        df,
        target="track_popularity",
    )

    # Semantic type remains numeric.
    assert result["type"] == "numeric"

    # ML task is regression.
    assert result["task_type"] == "regression"

    # Classification metadata must not be generated.
    assert result["class_count"] is None
    assert result["classes"] is None
    assert result["class_distribution"] == {}
    assert result["imbalance_ratio"] is None


def test_numeric_target_with_few_unique_values_is_classification():
    df = pd.DataFrame({
        "rating": [
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
        ]
    })

    result = check_target(
        df,
        target="rating",
    )

    # Semantic type remains numeric.
    assert result["type"] == "numeric"

    # Small discrete numeric target is classification.
    assert result["task_type"] == "classification"

    assert result["class_count"] == 3
    assert result["classes"] == 3
    assert result["imbalance_ratio"] is not None


def test_regression_target_does_not_calculate_fake_imbalance():
    df = pd.DataFrame({
        "score": list(range(1, 101))
    })

    result = check_target(
        df,
        target="score",
    )

    assert result["type"] == "numeric"
    assert result["task_type"] == "regression"

    # Regression must never receive class imbalance analysis.
    assert result["imbalance_ratio"] is None
    assert result["class_distribution"] == {}
    assert result["class_count"] is None
    assert result["classes"] is None


def test_numeric_target_with_exactly_20_unique_values_is_classification():
    df = pd.DataFrame({
        "target": list(range(20))
    })

    result = check_target(
        df,
        target="target",
    )

    assert result["type"] == "numeric"

    # Boundary condition:
    # 20 unique numeric values -> classification.
    assert result["task_type"] == "classification"

    assert result["class_count"] == 20
    assert result["classes"] == 20