import pandas as pd


def check_cardinality(df: pd.DataFrame) -> dict:
    """
    Analyze the number and percentage of unique values
    in every column.
    """

    result = {}

    total_rows = len(df)

    for column in df.columns:

        unique_count = int(
            df[column].nunique(dropna=True)
        )

        unique_percentage = (
            (unique_count / total_rows) * 100
            if total_rows > 0
            else 0.0
        )

        result[column] = {
            "unique_count": unique_count,
            "unique_percentage": round(
                unique_percentage,
                2
            ),
        }

    return result