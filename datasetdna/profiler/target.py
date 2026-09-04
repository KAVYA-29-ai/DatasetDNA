from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


# Numeric targets with more than this many unique values
# are treated as regression targets.
MAX_NUMERIC_CLASSIFICATION_CLASSES = 20


def _detect_target_task(
    series: pd.Series,
    detected_type: str,
) -> str:
    """
    Determine whether the target represents
    classification or regression.

    Rules:
        - numeric + <= 20 unique values -> classification
        - numeric + > 20 unique values  -> regression
        - boolean/categorical/date/id    -> classification
    """

    if detected_type == "numeric":
        unique_count = int(
            series.nunique(dropna=True)
        )

        if unique_count > MAX_NUMERIC_CLASSIFICATION_CLASSES:
            return "regression"

        return "classification"

    if detected_type in {
        "boolean",
        "categorical",
        "date",
        "id",
    }:
        return "classification"

    return "classification"


def check_target(
    df: pd.DataFrame,
    target: str | None = None,
) -> dict:
    """
    Analyze the target column.

    Semantic data type and ML task type are kept separate.

    Numeric target:
        <= 20 unique values -> classification
        > 20 unique values  -> regression

    Returns:
        provided
        column
        type
        task_type
        class_count
        classes
        class_distribution
        imbalance_ratio
        missing
        missing_count
        confidence
        is_id_like
    """

    # ---------------------------------------------------------
    # Target not provided
    # ---------------------------------------------------------

    if target is None:
        return {
            "provided": False,
            "column": None,
        }

    # ---------------------------------------------------------
    # Target does not exist
    # ---------------------------------------------------------

    if target not in df.columns:
        return {
            "provided": True,
            "column": target,
            "error": f"Target column '{target}' not found.",
        }

    # ---------------------------------------------------------
    # Semantic type detection
    # ---------------------------------------------------------

    types = check_types(df)

    type_info = types[target]

    detected_type = type_info["detected_type"]

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_count = int(
        df[target].isna().sum()
    )

    # ---------------------------------------------------------
    # Remove missing target values
    # ---------------------------------------------------------

    series = df[target].dropna()

    # ---------------------------------------------------------
    # Empty target
    # ---------------------------------------------------------

    if series.empty:
        return {
            "provided": True,
            "column": target,
            "type": "empty",
            "task_type": None,
            "classes": None,
            "class_count": 0,
            "class_distribution": {},
            "imbalance_ratio": None,
            "missing": missing_count,
            "missing_count": missing_count,
            "confidence": type_info.get("confidence"),
            "is_id_like": type_info.get(
                "is_id_like",
                False,
            ),
        }

    # ---------------------------------------------------------
    # Detect ML task
    # ---------------------------------------------------------

    task_type = _detect_target_task(
        series=series,
        detected_type=detected_type,
    )

    # ---------------------------------------------------------
    # Regression target
    # ---------------------------------------------------------

    if task_type == "regression":
        return {
            "provided": True,
            "column": target,

            # Semantic type.
            "type": detected_type,

            # ML task.
            "task_type": "regression",

            # Classification-only metadata.
            "classes": None,
            "class_count": None,
            "class_distribution": {},
            "imbalance_ratio": None,

            # Missing values.
            "missing": missing_count,
            "missing_count": missing_count,

            # Type detector metadata.
            "confidence": type_info.get("confidence"),
            "is_id_like": type_info.get(
                "is_id_like",
                False,
            ),
        }

    # ---------------------------------------------------------
    # Classification target
    # ---------------------------------------------------------

    value_counts = series.value_counts()

    total = len(series)

    distribution: dict[str, dict[str, int | float]] = {}

    for value, count in value_counts.items():
        percentage = (
            float(count)
            / total
            * 100
        )

        distribution[str(value)] = {
            "count": int(count),
            "percentage": round(
                percentage,
                2,
            ),
        }

    # ---------------------------------------------------------
    # Class count
    # ---------------------------------------------------------

    class_count = int(
        series.nunique(dropna=True)
    )

    # ---------------------------------------------------------
    # Imbalance ratio
    # ---------------------------------------------------------

    if len(value_counts) > 1:
        largest_class = int(
            value_counts.max()
        )

        smallest_class = int(
            value_counts.min()
        )

        if smallest_class > 0:
            imbalance_ratio = round(
                largest_class
                / smallest_class,
                2,
            )
        else:
            imbalance_ratio = None
    else:
        imbalance_ratio = None

    # ---------------------------------------------------------
    # Classification result
    # ---------------------------------------------------------

    return {
        "provided": True,
        "column": target,

        # Semantic type.
        "type": detected_type,

        # ML task.
        "task_type": "classification",

        # Classification metadata.
        "classes": class_count,
        "class_count": class_count,
        "class_distribution": distribution,
        "imbalance_ratio": imbalance_ratio,

        # Missing values.
        "missing": missing_count,
        "missing_count": missing_count,

        # Type detector metadata.
        "confidence": type_info.get("confidence"),
        "is_id_like": type_info.get(
            "is_id_like",
            False,
        ),
    }