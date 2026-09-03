from __future__ import annotations

import pandas as pd

from scipy.stats import chi2_contingency

from datasetdna.profiler.types import check_types


# =============================================================
# CONFIGURATION
# =============================================================

MAX_CATEGORICAL_CARDINALITY_PERCENT = 50.0


# =============================================================
# CRAMER'S V
# =============================================================

def _cramers_v(
    x: pd.Series,
    y: pd.Series,
) -> float | None:
    """
    Calculate bias-corrected Cramér's V between
    two categorical variables.
    """

    data = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    if data.empty:
        return None

    contingency_table = pd.crosstab(
        data["x"],
        data["y"],
    )

    # Both variables must contain at least
    # two categories.
    if (
        contingency_table.shape[0] < 2
        or contingency_table.shape[1] < 2
    ):
        return 0.0

    chi2 = chi2_contingency(
        contingency_table,
        correction=False,
    )[0]

    n = contingency_table.to_numpy().sum()

    if n <= 1:
        return 0.0

    phi2 = chi2 / n

    rows, columns = (
        contingency_table.shape
    )

    # ---------------------------------------------------------
    # Bias correction
    # ---------------------------------------------------------

    phi2_corrected = max(
        0,
        phi2
        - (
            (columns - 1)
            * (rows - 1)
        )
        / (n - 1),
    )

    rows_corrected = (
        rows
        - (
            (rows - 1) ** 2
            / (n - 1)
        )
    )

    columns_corrected = (
        columns
        - (
            (columns - 1) ** 2
            / (n - 1)
        )
    )

    denominator = min(
        columns_corrected - 1,
        rows_corrected - 1,
    )

    if denominator <= 0:
        return 0.0

    value = (
        phi2_corrected
        / denominator
    ) ** 0.5

    return round(
        float(value),
        4,
    )


# =============================================================
# CORRELATION PROFILER
# =============================================================

def check_correlations(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze relationships between meaningful variables.

    Numeric:
        Pearson correlation.

    Categorical:
        Bias-corrected Cramér's V.

    Identifier columns are excluded automatically using
    the centralized type detector.
    """

    result = {
        "numeric": [],
        "categorical": [],
    }

    # =========================================================
    # CENTRALIZED TYPE DETECTION
    # =========================================================

    types = check_types(df)

    # =========================================================
    # NUMERIC CORRELATIONS
    # =========================================================

    numerical_columns = [
        column
        for column in df.columns
        if types[column]["detected_type"]
        == "numeric"
    ]

    for i in range(
        len(numerical_columns)
    ):

        for j in range(
            i + 1,
            len(numerical_columns),
        ):

            column_1 = (
                numerical_columns[i]
            )

            column_2 = (
                numerical_columns[j]
            )

            pair = df[
                [
                    column_1,
                    column_2,
                ]
            ].dropna()

            if len(pair) < 2:
                continue

            # Ignore constant columns.
            if (
                pair[column_1].nunique()
                <= 1
                or pair[column_2].nunique()
                <= 1
            ):
                continue

            correlation = pair[
                column_1
            ].corr(
                pair[column_2],
                method="pearson",
            )

            if pd.isna(correlation):
                continue

            result["numeric"].append(
                {
                    "column_1": column_1,
                    "column_2": column_2,
                    "pearson": round(
                        float(correlation),
                        4,
                    ),
                }
            )

    # =========================================================
    # CATEGORICAL ASSOCIATIONS
    # =========================================================

    categorical_columns = [
        column
        for column in df.columns
        if types[column]["detected_type"]
        == "categorical"
    ]

    for i in range(
        len(categorical_columns)
    ):

        for j in range(
            i + 1,
            len(categorical_columns),
        ):

            column_1 = (
                categorical_columns[i]
            )

            column_2 = (
                categorical_columns[j]
            )

            values_1 = (
                df[column_1]
                .dropna()
            )

            values_2 = (
                df[column_2]
                .dropna()
            )

            if values_1.empty:
                continue

            if values_2.empty:
                continue

            # -------------------------------------------------
            # Cardinality protection
            # -------------------------------------------------
            #
            # Near-unique categorical columns such as:
            #
            #   email
            #   transaction_id
            #   UUID
            #
            # can produce misleadingly high Cramér's V.
            #
            # Skip the entire pair if either column has
            # more than 50% unique values.
            # -------------------------------------------------

            unique_pct_1 = (
                values_1.nunique()
                / len(values_1)
                * 100
            )

            unique_pct_2 = (
                values_2.nunique()
                / len(values_2)
                * 100
            )

            if (
                unique_pct_1
                > MAX_CATEGORICAL_CARDINALITY_PERCENT
                or unique_pct_2
                > MAX_CATEGORICAL_CARDINALITY_PERCENT
            ):
                continue

            value = _cramers_v(
                df[column_1],
                df[column_2],
            )

            if value is None:
                continue

            result["categorical"].append(
                {
                    "column_1": column_1,
                    "column_2": column_2,
                    "cramers_v": value,
                }
            )

    return result