from __future__ import annotations

import re

import pandas as pd


DATE_THRESHOLD = 0.90
BOOLEAN_THRESHOLD = 0.90
ID_THRESHOLD = 0.95

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

DATE_PATTERNS = [
    re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"),
    re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$"),
    re.compile(r"^\d{4}[-/]\d{1,2}$"),
    re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$"),
    re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$"),
]


def _looks_like_date(value: object) -> bool:
    """
    Return True when a value resembles a common date format.

    This prevents arbitrary text such as '3 A.M.' from being
    passed to pandas' date parser.
    """
    text = str(value).strip()

    if not text:
        return False

    return any(pattern.match(text) for pattern in DATE_PATTERNS)


def _parse_date_rate(series: pd.Series) -> float:
    """
    Estimate how strongly a string/object column represents dates.

    Only values matching known date-like patterns are sent to
    pandas' date parser. This avoids parsing arbitrary text such
    as '3 A.M.' as a datetime/timezone expression.
    """
    values = series.dropna()

    if values.empty:
        return 0.0

    candidate_mask = values.map(_looks_like_date)
    candidates = values[candidate_mask]

    if candidates.empty:
        return 0.0

    parsed = pd.to_datetime(
        candidates,
        errors="coerce",
        format="mixed",
    )

    return float(parsed.notna().mean()) * (
        len(candidates) / len(values)
    )


def _parse_boolean_rate(series: pd.Series) -> float:
    values = series.dropna()

    if values.empty:
        return 0.0

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    return float(
        normalized.isin(BOOLEAN_VALUES).mean()
    )


def _is_id_name(column: str) -> bool:
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


def detect_column_type(series: pd.Series) -> dict:
    column = str(series.name)
    values = series.dropna()

    if values.empty:
        return {
            "column": column,
            "detected_type": "empty",
            "confidence": 1.0,
            "is_id_like": False,
        }

    unique_percentage = values.nunique() / len(values)

    if pd.api.types.is_numeric_dtype(series):
        is_id = (
            unique_percentage >= ID_THRESHOLD
            and _is_id_name(column)
        )

        return {
            "column": column,
            "detected_type": "id" if is_id else "numeric",
            "confidence": 1.0,
            "is_id_like": is_id,
            "unique_percentage": round(
                float(unique_percentage),
                4,
            ),
        }

    if pd.api.types.is_bool_dtype(series):
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

    date_rate = _parse_date_rate(series)
    boolean_rate = _parse_boolean_rate(series)

    id_like = (
        unique_percentage >= ID_THRESHOLD
        and _is_id_name(column)
    )

    if id_like:
        detected_type = "id"
        confidence = unique_percentage

    elif boolean_rate >= BOOLEAN_THRESHOLD:
        detected_type = "boolean"
        confidence = boolean_rate

    elif date_rate >= DATE_THRESHOLD:
        detected_type = "date"
        confidence = date_rate

    else:
        detected_type = "categorical"
        confidence = 1.0

    return {
        "column": column,
        "detected_type": detected_type,
        "confidence": round(float(confidence), 4),
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


def check_types(df: pd.DataFrame) -> dict:
    return {
        column: detect_column_type(df[column])
        for column in df.columns
    }