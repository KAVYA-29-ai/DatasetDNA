from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


# =============================================================
# CONFIGURATION
# =============================================================

IQR_MULTIPLIER = 1.5


# =============================================================
# DOMAIN HELPERS
# =============================================================

NON_NEGATIVE_KEYWORDS = (
    "amount",
    "count",
    "quantity",
    "price",
    "cost",
    "income",
    "salary",
    "balance",
    "distance",
    "duration",
)


def _has_non_negative_domain(column: str) -> bool:
    column_lower = column.lower()

    # Age-like columns
    if (
        column_lower == "age"
        or column_lower.endswith("_age")
        or column_lower.startswith("age_")
    ):
        return True

    # Explicit non-negative quantities
    return any(
        keyword == column_lower
        or column_lower.startswith(f"{keyword}_")
        or column_lower.endswith(f"_{keyword}")
        for keyword in NON_NEGATIVE_KEYWORDS
    )


# =============================================================
# OUTLIER PROFILER
# =============================================================

def check_outliers(
    df: pd.DataFrame,
) -> dict:
    """
    Detect statistical outliers in meaningful numerical columns.

    Binary numeric columns containing only 0/1 values are excluded
    because they behave like categorical flags.

    Domain-aware lower bounds are applied to quantities that cannot
    logically be negative, such as age, count, price, income, etc.

    Columns detected as identifiers are excluded automatically.
    """

    result = {}

    types = check_types(df)

    for column in df.columns:

        column_type = types[column]

        # Only analyze columns classified as numeric.
        if column_type["detected_type"] != "numeric":
            continue

        # Binary numeric columns (0/1) are categorical flags,
        # not meaningful continuous numerical features.
        unique_values = df[column].dropna().unique()

        if len(unique_values) <= 2 and set(unique_values).issubset({0, 1}):
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
        # Domain-aware lower bound
        # -----------------------------------------------------

        if _has_non_negative_domain(column):
            lower_bound = max(
                0.0,
                float(lower_bound),
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