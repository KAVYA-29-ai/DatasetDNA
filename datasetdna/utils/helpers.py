from __future__ import annotations

import os

import pandas as pd


LARGE_FILE_SIZE_MB = 200
LARGE_FILE_SIZE_BYTES = LARGE_FILE_SIZE_MB * 1024 * 1024
LARGE_FILE_SAMPLE_SIZE = 100_000


MISSING_VALUES = [
    "",
    " ",
    "N/A",
    "N/a",
    "n/a",
    "NA",
    "na",
    "NULL",
    "Null",
    "null",
    "NONE",
    "None",
    "none",
    "?",
    "-",
    "unknown",
    "Unknown",
    "UNKNOWN",
]


def get_file_size_mb(path: str) -> float:
    return round(
        os.path.getsize(path) / (1024 * 1024),
        2,
    )


def is_large_file(path: str) -> bool:
    return os.path.getsize(path) > LARGE_FILE_SIZE_BYTES


def detect_encoding(path: str) -> str:
    with open(path, "rb") as file:
        raw_data = file.read(1024 * 1024)

    for encoding in ("utf-8", "utf-8-sig"):
        try:
            raw_data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    return "latin-1"


def detect_delimiter(
    path: str,
    encoding: str = "utf-8",
) -> str:
    try:
        sample = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding=encoding,
            nrows=5,
        )

        if len(sample.columns) <= 1:
            return ","

        with open(
            path,
            "r",
            encoding=encoding,
            errors="replace",
        ) as file:
            text = file.read(8192)

        lines = [
            line
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return ","

        first_line = lines[0]

        candidates = [
            ",",
            ";",
            "\t",
            "|",
        ]

        return max(
            candidates,
            key=lambda delimiter: first_line.count(delimiter),
        )

    except Exception:
        return ","


def normalize_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    for column in result.columns:
        if (
            pd.api.types.is_object_dtype(result[column])
            or pd.api.types.is_string_dtype(result[column])
        ):
            result[column] = (
                result[column]
                .astype("string")
                .str.strip()
            )

    result = result.replace(
        MISSING_VALUES,
        pd.NA,
    )

    return result


def _clean_numeric_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()

    if not text:
        return None

    text = (
        text
        .replace(",", "")
        .replace("$", "")
        .replace("₹", "")
        .replace("€", "")
        .replace("£", "")
        .replace("%", "")
    )

    try:
        return float(text)
    except ValueError:
        return None


def clean_numeric_like_columns(
    df: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    result = df.copy()

    for column in result.columns:
        series = result[column]

        # Already numeric — nothing to clean.
        if pd.api.types.is_numeric_dtype(series):
            continue

        non_null = series.dropna()

        # Completely missing column.
        if non_null.empty:
            continue

        converted = non_null.apply(
            _clean_numeric_value
        )

        success_rate = converted.notna().mean()

        # If the majority of non-null values are numeric-like,
        # convert the whole column. Invalid values become NaN.
        if success_rate >= threshold:
            result[column] = pd.to_numeric(
                result[column].apply(
                    _clean_numeric_value
                ),
                errors="coerce",
            )

    return result


def _load_large_csv(
    path: str,
    encoding: str,
    delimiter: str,
) -> pd.DataFrame:
    chunks = []

    for chunk in pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        engine="python",
        on_bad_lines="skip",
        keep_default_na=True,
        na_values=MISSING_VALUES,
        skipinitialspace=True,
        chunksize=LARGE_FILE_SAMPLE_SIZE,
    ):
        remaining = (
            LARGE_FILE_SAMPLE_SIZE
            - sum(len(item) for item in chunks)
        )

        if remaining <= 0:
            break

        chunks.append(
            chunk.head(remaining)
        )

        if (
            sum(len(item) for item in chunks)
            >= LARGE_FILE_SAMPLE_SIZE
        ):
            break

    if not chunks:
        return pd.DataFrame()

    return pd.concat(
        chunks,
        ignore_index=True,
    )


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    if not os.path.isfile(path):
        raise ValueError(
            f"Path is not a file: {path}"
        )

    if os.path.getsize(path) == 0:
        raise ValueError(
            "CSV file is empty."
        )

    encoding = detect_encoding(path)

    delimiter = detect_delimiter(
        path,
        encoding,
    )

    if is_large_file(path):
        df = _load_large_csv(
            path,
            encoding,
            delimiter,
        )

    else:
        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                encoding=encoding,
                engine="python",
                on_bad_lines="skip",
                keep_default_na=True,
                na_values=MISSING_VALUES,
                skipinitialspace=True,
            )

        except UnicodeDecodeError:
            encoding = "latin-1"

            df = pd.read_csv(
                path,
                sep=delimiter,
                encoding=encoding,
                engine="python",
                on_bad_lines="skip",
                keep_default_na=True,
                na_values=MISSING_VALUES,
                skipinitialspace=True,
            )

    df = normalize_missing_values(df)

    df = clean_numeric_like_columns(df)

    return df


def validate_dataframe(
    df: pd.DataFrame,
) -> None:
    if df is None:
        raise ValueError(
            "DataFrame cannot be None."
        )

    # Completely empty DataFrame:
    # zero columns AND zero rows.
    if len(df.columns) == 0 and len(df) == 0:
        raise ValueError(
            "CSV contains no data rows."
        )

    # DataFrame has rows but no columns.
    if len(df.columns) == 0:
        raise ValueError(
            "CSV contains no columns."
        )

    # Columns exist but there are no data rows.
    if len(df) == 0:
        raise ValueError(
            "CSV contains no data rows."
        )


def load_dataset(
    path: str,
) -> pd.DataFrame:
    df = load_csv(path)

    validate_dataframe(df)

    return df