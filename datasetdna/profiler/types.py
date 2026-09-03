from __future__ import annotations

import pandas as pd


# =============================================================
# CONFIGURATION
# =============================================================

DATE_THRESHOLD = 0.90
BOOLEAN_THRESHOLD = 0.90
ID_THRESHOLD = 0.95


# =============================================================
# BOOLEAN VALUES
# =============================================================

BOOLEAN_VALUES = {
    "yes",
    "no",
    "true",
    "false",
    "y",
    "n",
    "t",
    "f",
}


# =============================================================
# DATE DETECTION
# =============================================================

def _parse_date_rate(
    series: pd.Series,
) -> float:
    """
    Calculate the percentage of non-null values
    that can be parsed as dates.
    """

    values = series.dropna()

    if values.empty:
        return 0.0

    parsed = pd.to_datetime(
        values,
        errors="coerce",
        format="mixed",
    )

    return float(
        parsed.notna().mean()
    )


# =============================================================
# BOOLEAN DETECTION
# =============================================================

def _parse_boolean_rate(
    series: pd.Series,
) -> float:
    """
    Calculate the percentage of non-null values
    that look like boolean values.
    """

    values = series.dropna()

    if values.empty:
        return 0.0

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    return float(
        normalized.isin(
            BOOLEAN_VALUES
        ).mean()
    )


# =============================================================
# IDENTIFIER DETECTION
# =============================================================

def _is_id_name(
    column: str,
) -> bool:
    """
    Detect whether a column name strongly suggests
    that the column represents an identifier.
    """

    name = column.strip().lower()

    return (
        name == "id"
        or name.endswith("_id")
        or name.startswith("id_")
        or name in {
            "uuid",
            "email",
            "email_address",
            "customer_id",
            "user_id",
            "product_id",
            "transaction_id",
        }
    )


# =============================================================
# COLUMN TYPE DETECTION
# =============================================================

def detect_column_type(
    series: pd.Series,
) -> dict:
    """
    Detect the semantic type of a single column.

    Possible detected types:

        numeric
        categorical
        date
        boolean
        id
        empty

    The function also provides confidence and
    supporting detection statistics.
    """

    column = str(
        series.name
    )

    values = series.dropna()

    # ---------------------------------------------------------
    # EMPTY COLUMN
    # ---------------------------------------------------------

    if values.empty:

        return {
            "column": column,
            "detected_type": "empty",
            "confidence": 1.0,
            "is_id_like": False,
        }

    # ---------------------------------------------------------
    # CARDINALITY
    # ---------------------------------------------------------

    unique_percentage = (
        values.nunique()
        / len(values)
    )

    # ---------------------------------------------------------
    # NATIVE NUMERIC
    # ---------------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        series
    ):

        is_id = (
            unique_percentage >= ID_THRESHOLD
            and _is_id_name(column)
        )

        return {
            "column": column,
            "detected_type": (
                "id"
                if is_id
                else "numeric"
            ),
            "confidence": 1.0,
            "is_id_like": is_id,
            "unique_percentage": round(
                float(unique_percentage),
                4,
            ),
        }

    # ---------------------------------------------------------
    # NATIVE BOOLEAN
    # ---------------------------------------------------------

    if pd.api.types.is_bool_dtype(
        series
    ):

        return {
            "column": column,
            "detected_type": "boolean",
            "confidence": 1.0,
            "is_id_like": False,
            "unique_percentage": round(
                float(unique_percentage),
                4,
            ),
        }

    # ---------------------------------------------------------
    # STRING-LIKE ANALYSIS
    # ---------------------------------------------------------

    date_rate = _parse_date_rate(
        series
    )

    boolean_rate = _parse_boolean_rate(
        series
    )

    # ---------------------------------------------------------
    # ID DETECTION
    # ---------------------------------------------------------

    id_like = (
        unique_percentage >= ID_THRESHOLD
        and _is_id_name(column)
    )

    # ---------------------------------------------------------
    # TYPE DECISION
    # ---------------------------------------------------------

    if id_like:

        detected_type = "id"

        confidence = (
            unique_percentage
        )

    elif boolean_rate >= BOOLEAN_THRESHOLD:

        detected_type = "boolean"

        confidence = (
            boolean_rate
        )

    elif date_rate >= DATE_THRESHOLD:

        detected_type = "date"

        confidence = (
            date_rate
        )

    else:

        detected_type = "categorical"

        confidence = 1.0

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    return {
        "column": column,
        "detected_type": detected_type,
        "confidence": round(
            float(confidence),
            4,
        ),
        "is_id_like": id_like,
        "unique_percentage": round(
            float(unique_percentage),
            4,
        ),
        "date_success_rate": round(
            float(date_rate),
            4,
        ),
        "boolean_success_rate": round(
            float(boolean_rate),
            4,
        ),
    }


# =============================================================
# DATAFRAME TYPE PROFILER
# =============================================================

def check_types(
    df: pd.DataFrame,
) -> dict:
    """
    Detect the type of every column in a DataFrame.

    Returns:

        {
            "column_name": {
                ...
            }
        }
    """

    return {
        column: detect_column_type(
            df[column]
        )
        for column in df.columns
    }