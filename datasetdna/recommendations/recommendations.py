from __future__ import annotations


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

        # A dictionary representing one record.
        if all(
            not isinstance(item, (list, dict))
            for item in value.values()
        ):
            return [value]

        records: list[dict] = []

        for item in value.values():

            if isinstance(item, dict):

                records.append(
                    item
                )

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

    return _iter_records(
        signals
    )


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


# =============================================================
# MAIN RECOMMENDATION ENGINE
# =============================================================

def generate_recommendations(
    results: dict,
) -> list[dict]:
    """
    Generate deterministic, actionable recommendations
    from DatasetDNA profiler results.

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

        if percentage < 5:
            continue

        if percentage >= 20:

            severity = "high"

            action = (
                "Investigate why values are missing and "
                "consider an appropriate imputation or "
                "feature-removal strategy."
            )

        else:

            severity = "medium"

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

    if percentage <= 1:
        return

    if percentage > 10:

        severity = "high"

        action = (
            "Investigate the source of duplication and "
            "deduplicate records before model training."
        )

    else:

        severity = "medium"

        action = (
            "Review duplicate records and remove them "
            "if they represent repeated observations."
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

    if imbalance_ratio <= 2:
        return

    column = target.get(
        "column",
        "target",
    )

    if imbalance_ratio > 10:

        severity = "high"

        action = (
            "Consider class weights, resampling, or "
            "imbalance-aware evaluation metrics."
        )

    elif imbalance_ratio > 5:

        severity = "high"

        action = (
            "Consider class weights or resampling and "
            "evaluate the model with imbalance-aware metrics."
        )

    else:

        severity = "medium"

        action = (
            "Use stratified splitting and monitor class-aware "
            "evaluation metrics during model training."
        )

    recommendations.append(
        {
            "type": "target_imbalance",
            "severity": severity,
            "column": column,
            "message": (
                f"Target '{column}' has an imbalance ratio "
                f"of {imbalance_ratio:g}:1. "
                f"{action}"
            ),
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

        recommendations.append(
            {
                "type": "invalid_values",
                "severity": finding.get(
                    "severity",
                    "high",
                ),
                "column": column,
                "message": (
                    f"{column} contains suspicious values "
                    f"with observed range {value}. "
                    "Review and correct invalid observations "
                    "before model training."
                ),
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

        if value < 0.7:
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
            if value > 0.9
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
                    "Check whether the features are redundant "
                    "before model training."
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

        if percentage <= 1:
            continue

        column = signal.get(
            "column",
            "column",
        )

        if percentage > 10:

            severity = "high"

            action = (
                "Investigate whether these observations are "
                "data errors or legitimate extreme cases."
            )

        elif percentage > 5:

            severity = "medium"

            action = (
                "Review the extreme observations before "
                "deciding whether treatment is necessary."
            )

        else:

            severity = "low"

            action = (
                "Inspect the extreme observations and "
                "confirm that they are valid."
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

        if value <= 1:
            continue

        column = signal.get(
            "column",
            "column",
        )

        if value > 2:

            severity = "high"

            action = (
                "Consider a suitable transformation or "
                "robust modeling approach if the skew is "
                "problematic for the selected model."
            )

        else:

            severity = "medium"

            action = (
                "Inspect the distribution and consider a "
                "transformation if required by the model."
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
                    "Consider whether this categorical feature "
                    "should be encoded differently or excluded."
                ),
            }
        )