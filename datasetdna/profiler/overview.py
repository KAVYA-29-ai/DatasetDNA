import pandas as pd


def check_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 * 1024),
            2
        ),
    }
