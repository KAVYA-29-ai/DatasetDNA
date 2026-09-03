import pandas as pd


def check_schema(df: pd.DataFrame) -> dict:
    return {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }
