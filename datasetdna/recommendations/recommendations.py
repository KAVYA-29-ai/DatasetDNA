from __future__ import annotations


# =============================================================
# CONFIGURATION
# =============================================================

# Missing-value thresholds (%)
MISSING_RECOMMENDATION_THRESHOLD = 5
MISSING_HIGH_THRESHOLD = 20

# Duplicate-row thresholds (%)
DUPLICATE_RECOMMENDATION_THRESHOLD = 1
DUPLICATE_HIGH_THRESHOLD = 10

# Target imbalance thresholds (ratio)
TARGET_IMBALANCE_RECOMMENDATION_THRESHOLD = 2
TARGET_IMBALANCE_HIGH_THRESHOLD = 5

# Correlation thresholds
CORRELATION_RECOMMENDATION_THRESHOLD = 0.7
CORRELATION_HIGH_THRESHOLD = 0.9

# Outlier thresholds (%)
OUTLIER_RECOMMENDATION_THRESHOLD = 1
OUTLIER_MEDIUM_THRESHOLD = 5
OUTLIER_HIGH_THRESHOLD = 10

# Skewness thresholds
SKEWNESS_RECOMMENDATION_THRESHOLD = 1
SKEWNESS_HIGH_THRESHOLD = 2

# Mixed-type threshold
MIXED_TYPES_RECOMMENDATION_THRESHOLD = 2


# =============================================================
# HELPERS
# =============================================================

def _iter_records(value) -> list[dict]:
    """
    Normalize supported profiler/scorer structures into
    a list of dictionaries.

    Supported inputs include:

        list[dict]
        dict[str, dict]
        dict[str, list[dict]]
        dict representing a single record

    Invalid/non-dictionary values are ignored.
    """

    if isinstance(value, list):
        return [
            item
            for item in value
            if isinstance(item, dict)
        ]

    if isinstance(value, dict):

        # Dictionary representing one record.
        if all(
            not isinstance(item, (list, dict))
            for item in value.values()
        ):
            return [value]

        records: list[dict] = []

        for item in value.values():

            if isinstance(item, dict):
                records.append(item)

            elif isinstance(item, list):
                records.extend(
                    record
                    for record in item
                    if isinstance(record, dict)
                )

        return records

    return []


def _get_statistical_signals(
    health: dict,
) -> list[dict]:
    """
    Safely extract statistical signal records from
    the health-score structure.
    """

    statistical_signals = health.get(
        "statistical_signals",
        {},
    )

    if isinstance(
        statistical_signals,
        dict,
    ):

        signals = statistical_signals.get(
            "signals",
            statistical_signals,
        )

    else:

        signals = statistical_signals

    return _iter_records(signals)


def _get_quality_findings(
    health: dict,
) -> list[dict]:
    """
    Safely extract quality findings from the
    health-score structure.
    """

    quality_findings = health.get(
        "quality_findings",
        [],
    )

    return _iter_records(
        quality_findings
    )


def _get_column_type(
    results: dict,
    column: str,
) -> str | None:
    """
    Safely retrieve detected semantic/data type
    for a column.

    Supports both schema/type result structures
    without making recommendations depend on a
    single profiler representation.
    """

    # ---------------------------------------------------------
    # Direct type result
    # ---------------------------------------------------------

    types = results.get(
        "types",
        {},
    )

    if isinstance(types, dict):

        info = types.get(column)

        if isinstance(info, dict):

            detected_type = info.get(
                "detected_type"
            )

            if detected_type:
                return str(
                    detected_type
                ).lower()

    # ---------------------------------------------------------
    # Schema result
    # ---------------------------------------------------------

    schema = results.get(
        "schema",
        {},
    )

    if isinstance(schema, dict):

        info = schema.get(column)

        if isinstance(info, dict):

            detected_type = (
                info.get("detected_type")
                or info.get("type")
                or info.get("semantic_type")
            )

            if detected_type:
                return str(
                    detected_type
                ).lower()

    return None


def _get_column_cardinality(
    results: dict,
    column: str,
) -> dict:
    """
    Safely retrieve cardinality information.
    """

    cardinality = results.get(
        "cardinality",
        {},
    )

    if not isinstance(
        cardinality,
        dict,
    ):
        return {}

    info = cardinality.get(
        column,
        {},
    )

    return (
        info
        if isinstance(info, dict)
        else {}
    )


def _get_column_numerical_info(
    results: dict,
    column: str,
) -> dict:
    """
    Safely retrieve numerical statistics for a column.
    """

    numerical = results.get(
        "numerical",
        {},
    )

    if not isinstance(
        numerical,
        dict,
    ):
        return {}

    info = numerical.get(
        column,
        {},
    )

    return (
        info
        if isinstance(info, dict)
        else {}
    )


def _is_numeric_column(
    results: dict,
    column: str,
) -> bool:
    """
    Determine whether a column is numeric based on
    DatasetDNA's detected type or numerical profiler.
    """

    detected_type = _get_column_type(
        results,
        column,
    )

    if detected_type in {
        "numeric",
        "number",
        "float",
        "integer",
        "int",
    }:
        return True

    numerical_info = (
        _get_column_numerical_info(
            results,
            column,
        )
    )

    return bool(
        numerical_info
    )


def _is_categorical_column(
    results: dict,
    column: str,
) -> bool:
    """
    Determine whether a column is categorical.
    """

    detected_type = _get_column_type(
        results,
        column,
    )

    return detected_type in {
        "categorical",
        "category",
        "string",
        "object",
        "boolean",
    }


# =============================================================
# MAIN RECOMMENDATION ENGINE
# =============================================================

def generate_recommendations(
    results: dict,
) -> list[dict]:
    """
    Generate deterministic, contextual and actionable
    recommendations from DatasetDNA profiler results.

    This module does not calculate health scores.
    It only translates detected dataset issues/signals
    into actionable guidance.
    """

    recommendations: list[dict] = []

    _add_missing_recommendations(
        results,
        recommendations,
    )

    _add_duplicate_recommendations(
        results,
        recommendations,
    )

    _add_target_recommendations(
        results,
        recommendations,
    )

    _add_invalid_value_recommendations(
        results,
        recommendations,
    )

    _add_constant_column_recommendations(
        results,
        recommendations,
    )

    _add_correlation_recommendations(
        results,
        recommendations,
    )

    _add_outlier_recommendations(
        results,
        recommendations,
    )

    _add_skewness_recommendations(
        results,
        recommendations,
    )

    _add_cardinality_recommendations(
        results,
        recommendations,
    )

    _add_mixed_type_recommendations(
        results,
        recommendations,
    )

    _add_category_consistency_recommendations(
        results,
        recommendations,
    )

    return recommendations


# =============================================================
# MISSING VALUES
# =============================================================

def _add_missing_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    missing = results.get(
        "missing",
        {},
    )

    columns = missing.get(
        "columns",
        {},
    )

    if not isinstance(
        columns,
        dict,
    ):
        return

    for column, info in columns.items():

        if not isinstance(
            info,
            dict,
        ):
            continue

        percentage = float(
            info.get(
                "percentage",
                0,
            )
        )

        if percentage < MISSING_RECOMMENDATION_THRESHOLD:
            continue

        column_type = _get_column_type(
            results,
            column,
        )

        if percentage >= MISSING_HIGH_THRESHOLD:

            severity = "high"

            if column_type in {
                "numeric",
                "number",
                "float",
                "integer",
                "int",
            }:

                action = (
                    "Investigate the source of the missingness. "
                    "Because this is a numeric feature, compare "
                    "median/model-based imputation with feature "
                    "removal and validate the resulting distribution."
                )

            elif column_type in {
                "categorical",
                "category",
                "string",
                "object",
                "boolean",
            }:

                action = (
                    "Investigate why values are missing. "
                    "Consider an explicit missing category, "
                    "appropriate imputation, or feature removal "
                    "if the column has limited usable information."
                )

            else:

                action = (
                    "Investigate why values are missing and "
                    "consider an appropriate imputation or "
                    "feature-removal strategy."
                )

        else:

            severity = "medium"

            if column_type in {
                "numeric",
                "number",
                "float",
                "integer",
                "int",
            }:

                action = (
                    "Investigate the missingness and consider "
                    "median or model-based imputation. Validate "
                    "the distribution after imputation."
                )

            elif column_type in {
                "categorical",
                "category",
                "string",
                "object",
                "boolean",
            }:

                action = (
                    "Investigate the missingness and consider "
                    "an explicit missing category or suitable "
                    "categorical imputation strategy."
                )

            else:

                action = (
                    "Investigate the missingness and consider "
                    "an appropriate imputation strategy."
                )

        recommendations.append(
            {
                "type": "missing",
                "severity": severity,
                "column": column,
                "message": (
                    f"{column} has {percentage:g}% "
                    "missing values. "
                    f"{action}"
                ),
            }
        )


# =============================================================
# DUPLICATES
# =============================================================

def _add_duplicate_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    duplicates = results.get(
        "duplicates",
        {},
    )

    if not isinstance(
        duplicates,
        dict,
    ):
        return

    percentage = float(
        duplicates.get(
            "duplicate_percentage",
            0,
        )
    )

    if percentage <= DUPLICATE_RECOMMENDATION_THRESHOLD:
        return

    if percentage > DUPLICATE_HIGH_THRESHOLD:

        severity = "high"

        action = (
            "Investigate the source of duplication and "
            "deduplicate records before model training. "
            "Check whether repeated rows represent genuine "
            "observations or ingestion/pipeline errors."
        )

    else:

        severity = "medium"

        action = (
            "Review duplicate records and remove them if "
            "they represent repeated observations. If duplicates "
            "are legitimate repeated events, verify that they "
            "should remain before training."
        )

    recommendations.append(
        {
            "type": "duplicates",
            "severity": severity,
            "message": (
                f"{percentage:g}% of rows are duplicates. "
                f"{action}"
            ),
        }
    )


# =============================================================
# TARGET IMBALANCE
# =============================================================

def _add_target_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    target = results.get(
        "target",
        {},
    )

    if not isinstance(
        target,
        dict,
    ):
        return

    if not target.get(
        "provided",
        False,
    ):
        return

    imbalance_ratio = target.get(
        "imbalance_ratio"
    )

    if imbalance_ratio is None:
        return

    imbalance_ratio = float(
        imbalance_ratio
    )

    if (
        imbalance_ratio
        <= TARGET_IMBALANCE_RECOMMENDATION_THRESHOLD
    ):
        return

    column = target.get(
        "column",
        "target",
    )

    task_type = target.get(
        "task_type"
    )

    if imbalance_ratio > TARGET_IMBALANCE_HIGH_THRESHOLD:

        severity = "high"

        action = (
            "Consider class weights, carefully validated "
            "resampling, or imbalance-aware evaluation metrics. "
            "Use stratified splitting where appropriate and "
            "compare performance across classes."
        )

    else:

        severity = "medium"

        action = (
            "Use stratified splitting and monitor class-aware "
            "evaluation metrics such as precision, recall, F1, "
            "or balanced accuracy during model training."
        )

    if task_type == "classification":

        message = (
            f"Target '{column}' has an imbalance ratio "
            f"of {imbalance_ratio:g}:1. "
            f"{action}"
        )

    else:

        message = (
            f"Target '{column}' shows an imbalance ratio "
            f"of {imbalance_ratio:g}:1. "
            f"{action}"
        )

    recommendations.append(
        {
            "type": "target_imbalance",
            "severity": severity,
            "column": column,
            "message": message,
        }
    )


# =============================================================
# INVALID VALUES
# =============================================================

def _add_invalid_value_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    health = results.get(
        "health",
        {},
    )

    if not isinstance(
        health,
        dict,
    ):
        return

    quality = _get_quality_findings(
        health
    )

    for finding in quality:

        if finding.get(
            "type"
        ) != "invalid_values":
            continue

        column = finding.get(
            "column",
            "column",
        )

        value = finding.get(
            "value"
        )

        message = (
            f"{column} contains suspicious values "
            f"with observed range {value}. "
        )

        column_name = str(
            column
        ).strip().lower()

        if (
            "age" in column_name
            and (
                column_name == "age"
                or column_name.startswith("age_")
                or column_name.endswith("_age")
            )
        ):

            message += (
                "Review records outside the expected age range "
                "and determine whether they are data-entry errors "
                "before training."
            )

        elif any(
            keyword in column_name
            for keyword in (
                "income",
                "salary",
                "amount",
                "price",
                "cost",
                "quantity",
                "balance",
            )
        ):

            message += (
                "Because this column represents a quantity that "
                "is normally non-negative, inspect the affected "
                "records and determine whether they are data-entry "
                "errors, reversals, refunds, losses, or valid domain "
                "cases. Do not automatically replace them with zero."
            )

        else:

            message += (
                "Review and correct invalid observations "
                "before model training."
            )

        recommendations.append(
            {
                "type": "invalid_values",
                "severity": finding.get(
                    "severity",
                    "high",
                ),
                "column": column,
                "message": message,
            }
        )


# =============================================================
# CONSTANT COLUMNS
# =============================================================

def _add_constant_column_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    health = results.get(
        "health",
        {},
    )

    if not isinstance(
        health,
        dict,
    ):
        return

    quality = _get_quality_findings(
        health
    )

    for finding in quality:

        if finding.get(
            "type"
        ) != "constant":
            continue

        column = finding.get(
            "column",
            "column",
        )

        recommendations.append(
            {
                "type": "constant",
                "severity": finding.get(
                    "severity",
                    "medium",
                ),
                "column": column,
                "message": (
                    f"{column} has only one unique value. "
                    "Consider removing it because it provides "
                    "no useful variation for most ML models."
                ),
            }
        )


# =============================================================
# CORRELATIONS
# =============================================================

def _add_correlation_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    health = results.get(
        "health",
        {},
    )

    if not isinstance(
        health,
        dict,
    ):
        return

    signals = _get_statistical_signals(
        health
    )

    for signal in signals:

        if signal.get(
            "type"
        ) != "correlation":
            continue

        value = float(
            signal.get(
                "value",
                0,
            )
        )

        if value < CORRELATION_RECOMMENDATION_THRESHOLD:
            continue

        columns = signal.get(
            "columns",
            [],
        )

        if not isinstance(
            columns,
            list,
        ):
            continue

        if len(columns) != 2:
            continue

        severity = (
            "high"
            if value > CORRELATION_HIGH_THRESHOLD
            else "medium"
        )

        recommendations.append(
            {
                "type": "correlation",
                "severity": severity,
                "columns": columns,
                "message": (
                    f"{columns[0]} and {columns[1]} have a "
                    f"strong Pearson correlation of {value:.4f}. "
                    "Check whether the features are redundant, "
                    "derived from one another, or represent the "
                    "same underlying information before model "
                    "training."
                ),
            }
        )


# =============================================================
# OUTLIERS
# =============================================================

def _add_outlier_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    health = results.get(
        "health",
        {},
    )

    if not isinstance(
        health,
        dict,
    ):
        return

    signals = _get_statistical_signals(
        health
    )

    for signal in signals:

        if signal.get(
            "type"
        ) != "outliers":
            continue

        percentage = float(
            signal.get(
                "value",
                0,
            )
        )

        if percentage <= OUTLIER_RECOMMENDATION_THRESHOLD:
            continue

        column = signal.get(
            "column",
            "column",
        )

        if percentage > OUTLIER_HIGH_THRESHOLD:

            severity = "high"

            action = (
                "Investigate whether these observations are "
                "data errors or legitimate extreme cases. "
                "If valid, consider robust transformations or "
                "models rather than automatically deleting them."
            )

        elif percentage > OUTLIER_MEDIUM_THRESHOLD:

            severity = "medium"

            action = (
                "Review the extreme observations before deciding "
                "whether treatment is necessary. Confirm whether "
                "they are valid business observations or data errors."
            )

        else:

            severity = "low"

            action = (
                "Inspect the extreme observations and confirm "
                "that they are valid before applying any treatment."
            )

        recommendations.append(
            {
                "type": "outliers",
                "severity": severity,
                "column": column,
                "message": (
                    f"{column} contains {percentage:g}% "
                    f"potential outliers. {action}"
                ),
            }
        )


# =============================================================
# SKEWNESS
# =============================================================

def _add_skewness_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    health = results.get(
        "health",
        {},
    )

    if not isinstance(
        health,
        dict,
    ):
        return

    signals = _get_statistical_signals(
        health
    )

    for signal in signals:

        if signal.get(
            "type"
        ) != "skewness":
            continue

        value = float(
            signal.get(
                "value",
                0,
            )
        )

        if value <= SKEWNESS_RECOMMENDATION_THRESHOLD:
            continue

        column = signal.get(
            "column",
            "column",
        )

        if value > SKEWNESS_HIGH_THRESHOLD:

            severity = "high"

            action = (
                "Inspect the distribution for extreme concentration "
                "or long tails. Consider a suitable transformation "
                "or robust modeling approach if the skew is problematic "
                "for the selected model."
            )

        else:

            severity = "medium"

            action = (
                "Inspect the distribution and consider a transformation "
                "if required by the selected model."
            )

        recommendations.append(
            {
                "type": "skewness",
                "severity": severity,
                "column": column,
                "message": (
                    f"{column} has skewness of {value:.4f}. "
                    f"{action}"
                ),
            }
        )


# =============================================================
# CARDINALITY
# =============================================================

def _add_cardinality_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:

    health = results.get(
        "health",
        {},
    )

    if not isinstance(
        health,
        dict,
    ):
        return

    signals = _get_statistical_signals(
        health
    )

    for signal in signals:

        if signal.get(
            "type"
        ) != "cardinality":
            continue

        percentage = float(
            signal.get(
                "value",
                0,
            )
        )

        column = signal.get(
            "column",
            "column",
        )

        column_type = _get_column_type(
            results,
            column,
        )

        if column_type == "id":

            action = (
                "This appears to be an identifier-like column. "
                "Avoid one-hot encoding it as a normal categorical "
                "feature; exclude it unless it carries meaningful "
                "predictive information."
            )

        elif percentage >= 95:

            action = (
                "Review whether this feature behaves like an "
                "identifier. Avoid high-dimensional one-hot encoding "
                "unless the categories have meaningful predictive value."
            )

        else:

            action = (
                "Consider whether this categorical feature should "
                "be encoded differently, grouped into meaningful "
                "categories, or excluded."
            )

        recommendations.append(
            {
                "type": "cardinality",
                "severity": signal.get(
                    "severity",
                    "medium",
                ),
                "column": column,
                "message": (
                    f"{column} has {percentage:g}% "
                    "unique values among its observations. "
                    f"{action}"
                ),
            }
        )


# =============================================================
# MIXED TYPES
# =============================================================

def _add_mixed_type_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:
    """
    Generate recommendations for columns containing
    multiple Python value types.
    """

    mixed_types = results.get(
        "mixed_types",
        {},
    )

    if not isinstance(
        mixed_types,
        dict,
    ):
        return

    for column, info in mixed_types.items():

        if not isinstance(
            info,
            dict,
        ):
            continue

        types = info.get(
            "types",
            {},
        )

        if not isinstance(
            types,
            dict,
        ):
            continue

        if len(types) < MIXED_TYPES_RECOMMENDATION_THRESHOLD:
            continue

        type_names = ", ".join(
            str(type_name)
            for type_name in types.keys()
        )

        recommendations.append(
            {
                "type": "mixed_types",
                "severity": "medium",
                "column": column,
                "message": (
                    f"Column '{column}' contains mixed data "
                    f"types ({type_names}). Standardize the "
                    "values before model training."
                ),
                "value": type_names,
            }
        )


# =============================================================
# CATEGORY CONSISTENCY
# =============================================================

def _add_category_consistency_recommendations(
    results: dict,
    recommendations: list[dict],
) -> None:
    """
    Generate recommendations for categorical columns
    containing multiple representations of the same category.

    Example:

        Male / male / M
        Female / female / F
        Yes / yes / Y / True

    This is a normalization recommendation only.
    It does not affect the health score.
    """

    category_consistency = results.get(
        "category_consistency",
        {},
    )

    if not isinstance(
        category_consistency,
        dict,
    ):
        return

    for column, info in category_consistency.items():

        if not isinstance(
            info,
            dict,
        ):
            continue

        groups = info.get(
            "groups",
            {},
        )

        if not isinstance(
            groups,
            dict,
        ):
            continue

        if not groups:
            continue

        inconsistent_groups = []

        for canonical, values in groups.items():

            if not isinstance(
                values,
                (list, tuple, set),
            ):
                continue

            if len(values) < 2:
                continue

            formatted_values = ", ".join(
                str(value)
                for value in values
            )

            inconsistent_groups.append(
                f"{canonical}: {formatted_values}"
            )

        if not inconsistent_groups:
            continue

        group_text = "; ".join(
            inconsistent_groups
        )

        recommendations.append(
            {
                "type": "category_consistency",
                "severity": "medium",
                "column": column,
                "message": (
                    f"Column '{column}' contains multiple "
                    "representations of the same category. "
                    f"Detected groups: {group_text}. "
                    "Standardize categorical values before "
                    "model training."
                ),
                "value": group_text,
            }
        )