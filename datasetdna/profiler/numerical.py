import pandas as pd


def check_numerical(df: pd.DataFrame) -> dict:
    """
    Analyze all numerical columns in the dataset.

    Returns:
        dict: Statistical information for each numerical column.
    """

    result = {}

    numerical_df = df.select_dtypes(include="number")

    for column in numerical_df.columns:
        series = numerical_df[column].dropna()

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
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "skewness": round(float(skewness), 4)
            if pd.notna(skewness)
            else None,
        }

    return result