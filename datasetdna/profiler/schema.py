import pandas as pd


def check_schema(df: pd.DataFrame) -> dict:
    """
    Return the data type of every column in the dataset.
    """

    return {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }