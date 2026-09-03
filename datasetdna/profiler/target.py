from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


# =============================================================
# TARGET PROFILER
# =============================================================

def check_target(
    df: pd.DataFrame,
    target: str | None = None,
) -> dict:
    """
    Analyze the target column.

    Target type is determined using DatasetDNA's centralized
    semantic type detector.

    Supported semantic target types include:

        numeric
        categorical
        boolean
        date
        id
        empty
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
            "error": (
                f"Target column '{target}' "
                f"not found."
            ),
        }

    # ---------------------------------------------------------
    # Semantic type detection
    # ---------------------------------------------------------

    types = check_types(
        df
    )

    type_info = types[
        target
    ]

    detected_type = type_info[
        "detected_type"
    ]

    # ---------------------------------------------------------
    # Remove missing values
    # ---------------------------------------------------------

    series = df[
        target
    ].dropna()

    # ---------------------------------------------------------
    # Empty target
    # ---------------------------------------------------------

    if series.empty:

        return {
            "provided": True,
            "column": target,
            "type": "empty",
            "class_count": 0,
            "class_distribution": {},
            "imbalance_ratio": None,
        }

    # ---------------------------------------------------------
    # Class distribution
    # ---------------------------------------------------------

    value_counts = series.value_counts()

    total = len(
        series
    )

    distribution = {}

    for value, count in value_counts.items():

        percentage = (
            count
            / total
            * 100
        )

        distribution[str(value)] = {
            "count": int(
                count
            ),
            "percentage": round(
                float(percentage),
                2,
            ),
        }

    # ---------------------------------------------------------
    # Imbalance ratio
    # ---------------------------------------------------------

    if len(value_counts) > 1:

        largest_class = (
            value_counts.max()
        )

        smallest_class = (
            value_counts.min()
        )

        if smallest_class > 0:

            imbalance_ratio = round(
                float(
                    largest_class
                    / smallest_class
                ),
                2,
            )

        else:

            imbalance_ratio = None

    else:

        imbalance_ratio = None

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    return {
        "provided": True,
        "column": target,

        # Semantic type instead of raw pandas dtype.
        "type": detected_type,

        "class_count": int(
            series.nunique()
        ),

        "class_distribution": distribution,

        "imbalance_ratio": imbalance_ratio,

        # Keep detector metadata available for
        # downstream components.
        "confidence": type_info.get(
            "confidence"
        ),

        "is_id_like": type_info.get(
            "is_id_like",
            False,
        ),
    }