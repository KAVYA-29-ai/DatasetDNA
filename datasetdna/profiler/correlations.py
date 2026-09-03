import pandas as pd
from scipy.stats import chi2_contingency


def _cramers_v(
    x: pd.Series,
    y: pd.Series
) -> float | None:
    """
    Calculate Cramér's V between two categorical variables.
    """

    data = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    if data.empty:
        return None

    contingency_table = pd.crosstab(
        data["x"],
        data["y"]
    )

    if contingency_table.shape[0] < 2:
        return 0.0

    if contingency_table.shape[1] < 2:
        return 0.0

    chi2 = chi2_contingency(
        contingency_table,
        correction=False
    )[0]

    n = contingency_table.to_numpy().sum()

    if n <= 1:
        return 0.0

    phi2 = chi2 / n

    rows, columns = contingency_table.shape

    phi2_corrected = max(
        0,
        phi2 -
        ((columns - 1) * (rows - 1)) / (n - 1)
    )

    rows_corrected = (
        rows -
        ((rows - 1) ** 2 / (n - 1))
    )

    columns_corrected = (
        columns -
        ((columns - 1) ** 2 / (n - 1))
    )

    denominator = min(
        columns_corrected - 1,
        rows_corrected - 1
    )

    if denominator <= 0:
        return 0.0

    value = (
        phi2_corrected / denominator
    ) ** 0.5

    return round(float(value), 4)


def check_correlations(df: pd.DataFrame) -> dict:
    """
    Calculate correlations and associations.

    This function ONLY returns raw statistical results.
    It does not decide whether a correlation is high,
    suspicious, or problematic.
    """

    result = {
        "numeric": [],
        "categorical": [],
    }

    # =================================
    # NUMERICAL - PEARSON
    # =================================

    numerical_df = df.select_dtypes(
        include="number"
    )

    columns = list(numerical_df.columns)

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            column_1 = columns[i]
            column_2 = columns[j]

            pair = numerical_df[
                [column_1, column_2]
            ].dropna()

            if len(pair) < 2:
                continue

            correlation = pair[
                column_1
            ].corr(
                pair[column_2],
                method="pearson"
            )

            if pd.isna(correlation):
                continue

            result["numeric"].append({
                "column_1": column_1,
                "column_2": column_2,
                "pearson": round(
                    float(correlation),
                    4
                ),
            })

    # =================================
    # CATEGORICAL - CRAMÉR'S V
    # =================================

    categorical_df = df.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    )

    categorical_columns = list(
        categorical_df.columns
    )

    for i in range(
        len(categorical_columns)
    ):

        for j in range(
            i + 1,
            len(categorical_columns)
        ):

            column_1 = categorical_columns[i]
            column_2 = categorical_columns[j]

            value = _cramers_v(
                categorical_df[column_1],
                categorical_df[column_2]
            )

            if value is None:
                continue

            result["categorical"].append({
                "column_1": column_1,
                "column_2": column_2,
                "cramers_v": value,
            })

    return result