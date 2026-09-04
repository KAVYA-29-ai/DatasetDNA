from __future__ import annotations

import pandas as pd


def _python_type_name(value) -> str:
    """Return a readable Python type name."""
    return type(value).__name__


def check_mixed_types(df: pd.DataFrame) -> dict:
    """
    Detect columns containing multiple Python value types.

    Only non-null values are considered.

    Returns:
        A dictionary containing mixed-type information for each
        affected column.
    """

    results = {}

    for column in df.columns:
        series = df[column].dropna()

        if series.empty:
            continue

        type_counts = (
            series
            .map(_python_type_name)
            .value_counts()
            .to_dict()
        )

        if len(type_counts) <= 1:
            continue

        results[column] = {
            "types": type_counts,
            "type_count": len(type_counts),
            "message": (
                f"Column '{column}' contains multiple "
                "Python value types."
            ),
        }

    return results