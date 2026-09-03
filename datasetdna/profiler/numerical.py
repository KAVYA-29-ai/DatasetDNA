from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


def check_numerical(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze numerical columns while excluding columns
    detected as identifiers.

    Identifier columns such as customer_id should not be
    treated as meaningful numerical features.
    """

    result = {}

    types = check_types(df)

    for column in df.columns:

        column_type = types[column]

        if column_type["detected_type"] != "numeric":
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