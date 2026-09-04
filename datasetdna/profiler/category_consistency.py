from __future__ import annotations

import re

import pandas as pd


# Known semantic aliases where different representations
# commonly refer to the same category.
CATEGORY_ALIASES = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female",
    "yes": "yes",
    "y": "yes",
    "true": "yes",
    "no": "no",
    "n": "no",
    "false": "no",
}


def _normalize_value(value) -> str:
    """
    Normalize a categorical value for consistency comparison.

    Normalization is intentionally conservative:
    - convert to string
    - strip surrounding whitespace
    - lowercase
    - collapse repeated whitespace
    """

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    return value


def _canonical_category(value: str) -> str:
    """
    Return a canonical representation for known category aliases.
    """

    return CATEGORY_ALIASES.get(
        value,
        value,
    )


def check_category_consistency(
    df: pd.DataFrame,
) -> dict:
    """
    Detect categorical columns containing multiple
    representations of the same semantic category.

    Example:

        Male
        male
        M

    is reported as an inconsistency because all three
    representations map to the same canonical category.

    Returns:
        Dictionary containing findings for affected columns.
    """

    results: dict = {}

    for column in df.columns:

        series = df[column].dropna()

        if series.empty:
            continue

        # Category consistency is intended for textual/categorical
        # columns, not continuous numerical features.
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue

        original_values = {
            str(value).strip()
            for value in series
        }

        if len(original_values) <= 1:
            continue

        normalized_groups: dict[str, set[str]] = {}

        for value in original_values:

            normalized = _normalize_value(
                value
            )

            canonical = _canonical_category(
                normalized
            )

            normalized_groups.setdefault(
                canonical,
                set(),
            ).add(value)

        inconsistent_groups = {
            canonical: sorted(values)
            for canonical, values in normalized_groups.items()
            if len(values) > 1
        }

        if not inconsistent_groups:
            continue

        results[column] = {
            "groups": inconsistent_groups,
            "group_count": len(
                inconsistent_groups
            ),
            "message": (
                f"Column '{column}' contains multiple "
                "representations of the same category."
            ),
        }

    return results