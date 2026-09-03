import pandas as pd


IQR_MULTIPLIER = 1.5


def check_outliers(df: pd.DataFrame) -> dict:
    """
    Detect numerical outliers using the IQR method.

    Returns:
        dict: Q1, Q3, IQR, bounds and outlier count for each
        numerical column.
    """

    result = {}

    numerical_df = df.select_dtypes(include="number")

    for column in numerical_df.columns:

        series = numerical_df[column].dropna()

        if series.empty:
            result[column] = {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "lower_bound": None,
                "upper_bound": None,
            }
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (IQR_MULTIPLIER * iqr)
        upper_bound = q3 + (IQR_MULTIPLIER * iqr)

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        outlier_count = len(outliers)

        outlier_percentage = (
            (outlier_count / len(series)) * 100
            if len(series) > 0
            else 0.0
        )

        result[column] = {
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "outlier_count": int(outlier_count),
            "outlier_percentage": round(
                float(outlier_percentage),
                2
            ),
        }

    return result