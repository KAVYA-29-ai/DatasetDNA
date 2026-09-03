import pandas as pd


def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Check for exact duplicate rows in the dataset.

    Returns:
        dict: Duplicate row count and percentage.
    """

    duplicate_count = int(df.duplicated().sum())

    total_rows = len(df)

    duplicate_percentage = (
        (duplicate_count / total_rows) * 100
        if total_rows > 0
        else 0.0
    )

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(duplicate_percentage, 2),
    }