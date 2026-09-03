from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


def check_categorical(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze columns detected as categorical.

    Columns detected as:
        - id
        - numeric
        - date
        - boolean
        - empty

    are excluded automatically.
    """

    result = {}

    types = check_types(df)

    for column in df.columns:

        # -----------------------------------------------------
        # Only analyze semantic categorical columns
        # -----------------------------------------------------

        if (
            types[column]["detected_type"]
            != "categorical"
        ):
            continue

        series = df[column].dropna()

        # -----------------------------------------------------
        # Empty column
        # -----------------------------------------------------

        if series.empty:

            result[column] = {
                "unique_count": 0,
                "total_values": 0,
                "categories": {},
            }

            continue

        # -----------------------------------------------------
        # Category frequencies
        # -----------------------------------------------------

        value_counts = series.value_counts()

        total_values = len(
            series
        )

        categories = {}

        for category, count in value_counts.items():

            percentage = (
                count
                / total_values
                * 100
            )

            categories[str(category)] = {
                "count": int(count),
                "percentage": round(
                    float(percentage),
                    2,
                ),
            }

        # -----------------------------------------------------
        # Result
        # -----------------------------------------------------

        result[column] = {
            "unique_count": int(
                series.nunique()
            ),
            "total_values": int(
                total_values
            ),
            "categories": categories,
        }

    return result