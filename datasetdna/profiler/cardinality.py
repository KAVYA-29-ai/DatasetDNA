from __future__ import annotations

import pandas as pd

from datasetdna.profiler.types import check_types


def check_cardinality(
    df: pd.DataFrame,
) -> dict:
    """
    Analyze column cardinality.

    Cardinality is calculated for every column, but identifier
    columns are marked as ID-like so downstream scoring can
    distinguish expected high cardinality from suspicious
    high cardinality.

    Identifier columns are NOT removed from the raw cardinality
    report because their uniqueness is still useful information.
    """

    result = {}

    types = check_types(df)

    total_rows = len(df)

    for column in df.columns:

        unique_count = int(
            df[column].nunique(
                dropna=True
            )
        )

        unique_percentage = (
            (
                unique_count
                / total_rows
            )
            * 100
            if total_rows > 0
            else 0.0
        )

        column_type = types[column]

        result[column] = {
            "unique_count": unique_count,
            "unique_percentage": round(
                float(unique_percentage),
                2,
            ),
            "detected_type": column_type[
                "detected_type"
            ],
            "is_id_like": column_type[
                "is_id_like"
            ],
        }

    return result