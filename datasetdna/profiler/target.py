import pandas as pd


def check_target(
    df: pd.DataFrame,
    target: str | None = None
) -> dict:
    """
    Analyze the target column.

    For categorical targets:
        - class distribution
        - class percentages
        - imbalance ratio

    Returns:
        dict containing target analysis.
    """

    if target is None:
        return {
            "provided": False,
            "column": None,
        }

    if target not in df.columns:
        return {
            "provided": True,
            "column": target,
            "error": f"Target column '{target}' not found.",
        }

    series = df[target].dropna()

    if series.empty:
        return {
            "provided": True,
            "column": target,
            "type": "empty",
            "class_distribution": {},
            "imbalance_ratio": None,
        }

    value_counts = series.value_counts()

    total = len(series)

    distribution = {}

    for value, count in value_counts.items():

        percentage = (count / total) * 100

        distribution[str(value)] = {
            "count": int(count),
            "percentage": round(float(percentage), 2),
        }

    if len(value_counts) > 1:

        largest_class = value_counts.max()
        smallest_class = value_counts.min()

        imbalance_ratio = (
            largest_class / smallest_class
        )

        imbalance_ratio = round(
            float(imbalance_ratio),
            2
        )

    else:
        imbalance_ratio = None

    return {
        "provided": True,
        "column": target,
        "type": str(df[target].dtype),
        "class_count": int(series.nunique()),
        "class_distribution": distribution,
        "imbalance_ratio": imbalance_ratio,
    }