from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


def check_categorical(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze categorical columns and binary numeric columns.

    Columns detected as:
        - categorical
        - binary numeric (0/1)

    are analyzed.

    Columns detected as:
        - id
        - numeric (non-binary)
        - date
        - boolean
        - empty

    are excluded automatically.
    """

    result = {}

    types = check_types(df)

    for column in df.columns:

        detected_type = types[column]["detected_type"]

        # -----------------------------------------------------
        # Determine whether column should be analyzed
        # -----------------------------------------------------

        if detected_type == "categorical":
            should_analyze = True

        elif detected_type == "numeric":
            # Binary numeric columns such as 0/1 behave like
            # categorical flags and should have frequency analysis.
            unique_values = df[column].dropna().unique()

            should_analyze = (
                len(unique_values) <= 2
                and set(unique_values).issubset({0, 1})
            )

        else:
            should_analyze = False

        if not should_analyze:
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

        total_values = len(series)

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