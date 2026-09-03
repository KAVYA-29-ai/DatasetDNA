import pandas as pd


def check_missing(df: pd.DataFrame) -> dict:
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    return {
        "total_missing_cells": int(missing_count.sum()),
        "columns": {
            column: {
                "count": int(missing_count[column]),
                "percentage": round(float(missing_pct[column]), 2),
            }
            for column in df.columns
        },
    }
