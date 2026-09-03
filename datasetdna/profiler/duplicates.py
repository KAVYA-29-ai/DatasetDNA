from __future__ import annotations

import pandas as pd


# =============================================================
# IDENTIFIER DETECTION
# =============================================================

ID_NAME_HINTS = {
    "id",
    "uuid",
    "email",
    "email_address",
    "customer_id",
    "user_id",
    "product_id",
    "transaction_id",
}


def _is_id_column(
    column: str,
) -> bool:
    """
    Detect columns that represent record identifiers.

    Identifier columns should not determine whether two
    records are duplicates because IDs are expected to differ.
    """

    name = column.strip().lower()

    return (
        name in ID_NAME_HINTS
        or name.endswith("_id")
        or name.startswith("id_")
    )


# =============================================================
# VALUE NORMALIZATION
# =============================================================

def _normalize_value(value):
    """
    Normalize a single cell for duplicate comparison.

    Handles:
    - surrounding whitespace
    - numeric strings
    - formatted numbers
    - missing values
    - case differences
    """

    if pd.isna(value):
        return None

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return None

        # Normalize numeric formatting.
        cleaned = (
            value
            .replace(",", "")
            .replace("$", "")
            .replace("₹", "")
            .replace("€", "")
            .replace("£", "")
        )

        try:
            return float(cleaned)

        except ValueError:
            return value.casefold()

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    return value


# =============================================================
# DUPLICATE COMPARISON DATA
# =============================================================

def _normalize_for_duplicate_check(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a canonical DataFrame for duplicate detection.

    Identifier columns are intentionally excluded because
    different identifiers can belong to otherwise identical
    records.
    """

    comparison_columns = [
        column
        for column in df.columns
        if not _is_id_column(column)
    ]

    if not comparison_columns:
        return pd.DataFrame(
            index=df.index
        )

    normalized = df[
        comparison_columns
    ].copy()

    for column in normalized.columns:

        normalized[column] = normalized[
            column
        ].map(
            _normalize_value
        )

    return normalized


# =============================================================
# PUBLIC API
# =============================================================

def check_duplicates(
    df: pd.DataFrame,
) -> dict:
    """
    Detect duplicate records while ignoring identifier columns.

    Example:

        customer_id | name
        ------------|------
        1001        | Rahul
        1013        | Rahul

    These are considered duplicates if every non-ID field matches.
    """

    normalized = _normalize_for_duplicate_check(
        df
    )

    duplicate_count = int(
        normalized.duplicated(
            keep="first"
        ).sum()
    )

    total_rows = len(
        normalized
    )

    duplicate_percentage = (
        duplicate_count
        / total_rows
        * 100
        if total_rows > 0
        else 0.0
    )

    ignored_columns = [
        column
        for column in df.columns
        if _is_id_column(column)
    ]

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(
            duplicate_percentage,
            2,
        ),
        "ignored_identifier_columns": ignored_columns,
    }