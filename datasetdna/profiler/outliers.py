from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


# =============================================================
# CONFIGURATION
# =============================================================

IQR_MULTIPLIER = 1.5


# =============================================================
# OUTLIER PROFILER
# =============================================================

def check_outliers(
    df: pd.DataFrame,
) -> dict:
    """
    Detect statistical outliers in meaningful numerical columns.

    Columns detected as identifiers are excluded automatically.

    Example:

        customer_id      -> excluded
        age              -> analyzed
        salary           -> analyzed
        purchase_amount  -> analyzed
    """

    result = {}

    types = check_types(df)

    for column in df.columns:

        column_type = types[column]

        # Only analyze columns classified as numeric.
        if column_type["detected_type"] != "numeric":
            continue

        series = df[column].dropna()

        # -----------------------------------------------------
        # Empty column
        # -----------------------------------------------------

        if series.empty:

            result[column] = {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "lower_bound": None,
                "upper_bound": None,
            }

            continue

        # -----------------------------------------------------
        # Quartiles
        # -----------------------------------------------------

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = (
            q1
            - (
                IQR_MULTIPLIER
                * iqr
            )
        )

        upper_bound = (
            q3
            + (
                IQR_MULTIPLIER
                * iqr
            )
        )

        # -----------------------------------------------------
        # Detect outliers
        # -----------------------------------------------------

        outliers = series[
            (series < lower_bound)
            | (series > upper_bound)
        ]

        outlier_count = len(
            outliers
        )

        outlier_percentage = (
            (
                outlier_count
                / len(series)
            )
            * 100
            if len(series) > 0
            else 0.0
        )

        # -----------------------------------------------------
        # Result
        # -----------------------------------------------------

        result[column] = {
            "q1": round(
                float(q1),
                4,
            ),
            "q3": round(
                float(q3),
                4,
            ),
            "iqr": round(
                float(iqr),
                4,
            ),
            "lower_bound": round(
                float(lower_bound),
                4,
            ),
            "upper_bound": round(
                float(upper_bound),
                4,
            ),
            "outlier_count": int(
                outlier_count
            ),
            "outlier_percentage": round(
                float(outlier_percentage),
                2,
            ),
        }

    return result