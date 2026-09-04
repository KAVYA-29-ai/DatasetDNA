from __future__ import annotations

import pandas as pd

from datasetdna.profiler.overview import check_overview
from datasetdna.profiler.schema import check_schema
from datasetdna.profiler.mixed_types import check_mixed_types
from datasetdna.profiler.category_consistency import (
    check_category_consistency,
)
from datasetdna.profiler.missing import check_missing
from datasetdna.profiler.duplicates import check_duplicates
from datasetdna.profiler.cardinality import check_cardinality
from datasetdna.profiler.numerical import check_numerical
from datasetdna.profiler.categorical import check_categorical
from datasetdna.profiler.outliers import check_outliers
from datasetdna.profiler.correlations import check_correlations
from datasetdna.profiler.target import check_target

from datasetdna.scoring.health_score import calculate_health_score

from datasetdna.recommendations.recommendations import (
    generate_recommendations,
)


TARGET_COLUMN_CANDIDATES = (
    "target",
    "label",
    "churn",
)


def infer_target(
    df: pd.DataFrame,
) -> tuple[str | None, bool]:
    """
    Infer a target column when one is not explicitly provided.

    Priority:
        target -> label -> churn

    Returns:
        (column_name, inferred)
    """

    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in normalized_columns:
            return normalized_columns[candidate], True

    return None, False


def profile_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
) -> dict:
    """
    Run the complete DatasetDNA profiling pipeline
    directly on an in-memory pandas DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "DatasetDNA expects a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot profile an empty DataFrame."
        )

    target_was_inferred = False

    if target is None:
        target, target_was_inferred = infer_target(df)

    overview = check_overview(df)
    schema = check_schema(df)
    mixed_types = check_mixed_types(df)
    category_consistency = check_category_consistency(df)
    missing = check_missing(df)
    duplicates = check_duplicates(df)
    cardinality = check_cardinality(df)
    numerical = check_numerical(df)
    categorical = check_categorical(df)
    outliers = check_outliers(df)
    correlations = check_correlations(df)

    target_result = check_target(
        df,
        target,
    )

    if target_was_inferred:
        target_result["inferred"] = True

    results = {
        "overview": overview,
        "schema": schema,
        "mixed_types": mixed_types,
        "category_consistency": category_consistency,
        "missing": missing,
        "duplicates": duplicates,
        "cardinality": cardinality,
        "numerical": numerical,
        "categorical": categorical,
        "outliers": outliers,
        "correlations": correlations,
        "target": target_result,
    }

    health = calculate_health_score(results)

    results["health"] = health

    recommendations = generate_recommendations(
        results
    )

    results["recommendations"] = recommendations

    return results