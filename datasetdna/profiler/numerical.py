from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


def check_numerical(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze meaningful numerical columns.

    Binary numeric columns containing only 0/1 values are excluded
    because they behave like categorical flags rather than continuous
    numerical features.

    Identifier columns are also excluded based on type detection.
    """

    result = {}

    types = check_types(df)

    for column in df.columns:

        column_type = types[column]

        if column_type["detected_type"] != "numeric":
            continue

        # Binary numeric columns (0/1) are categorical flags,
        # not meaningful continuous numerical features.
        unique_values = df[column].dropna().unique()

        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1}):
            continue

        series = df[column].dropna()

        if series.empty:
            result[column] = {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "skewness": None,
            }

            continue

        skewness = series.skew()

        result[column] = {
            "count": int(series.count()),
            "mean": round(
                float(series.mean()),
                4,
            ),
            "median": round(
                float(series.median()),
                4,
            ),
            "std": round(
                float(series.std()),
                4,
            ),
            "min": round(
                float(series.min()),
                4,
            ),
            "max": round(
                float(series.max()),
                4,
            ),
            "skewness": (
                round(
                    float(skewness),
                    4,
                )
                if pd.notna(skewness)
                else None
            ),
        }

    return result