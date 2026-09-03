from __future__ import annotations

import pandas as pd


# =============================================================
# QUALITY FINDINGS
# =============================================================

def _get_quality_findings(
    results: dict,
) -> list[dict]:

    findings = []

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing = results.get(
        "missing",
        {},
    )

    for column, info in missing.get(
        "columns",
        {},
    ).items():

        percentage = float(
            info.get(
                "percentage",
                0,
            )
        )

        if percentage >= 5:

            if percentage > 50:
                severity = "high"

            elif percentage >= 20:
                severity = "high"

            else:
                severity = "medium"

            findings.append({
                "type": "missing",
                "severity": severity,
                "message": (
                    f"Column '{column}' contains "
                    f"{percentage}% missing values."
                ),
                "column": column,
                "value": percentage,
            })

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    duplicates = results.get(
        "duplicates",
        {},
    )

    duplicate_percentage = float(
        duplicates.get(
            "duplicate_percentage",
            0,
        )
    )

    if duplicate_percentage > 1:

        severity = (
            "high"
            if duplicate_percentage > 10
            else "medium"
        )

        findings.append({
            "type": "duplicates",
            "severity": severity,
            "message": (
                f"Dataset contains "
                f"{duplicate_percentage}% duplicate rows."
            ),
            "value": duplicate_percentage,
        })

    # ---------------------------------------------------------
    # Target imbalance
    # ---------------------------------------------------------

    target = results.get(
        "target",
        {},
    )

    imbalance_ratio = target.get(
        "imbalance_ratio"
    )

    if imbalance_ratio is not None:

        imbalance_ratio = float(
            imbalance_ratio
        )

        if imbalance_ratio > 2:

            if imbalance_ratio > 10:
                severity = "high"

            elif imbalance_ratio > 5:
                severity = "high"

            else:
                severity = "medium"

            findings.append({
                "type": "target_imbalance",
                "severity": severity,
                "message": (
                    f"Target classes have an "
                    f"imbalance ratio of "
                    f"{imbalance_ratio}:1."
                ),
                "value": imbalance_ratio,
            })

    # ---------------------------------------------------------
    # Constant columns
    # ---------------------------------------------------------

    cardinality = results.get(
        "cardinality",
        {},
    )

    for column, info in cardinality.items():

        unique_count = int(
            info.get(
                "unique_count",
                0,
            )
        )

        if unique_count == 1:

            findings.append({
                "type": "constant",
                "severity": "medium",
                "message": (
                    f"Column '{column}' "
                    f"is constant."
                ),
                "column": column,
                "value": unique_count,
            })

    # ---------------------------------------------------------
    # Suspicious / invalid values
    # ---------------------------------------------------------

    numerical = results.get(
        "numerical",
        {},
    )

    percentage_keywords = (
        "percentage",
        "percent",
        "pct",
        "rate",
    )

    probability_keywords = (
        "probability",
        "prob",
        "confidence",
    )

    non_negative_keywords = (
        "amount",
        "count",
        "quantity",
        "price",
        "cost",
        "income",
        "salary",
        "balance",
        "distance",
        "duration",
    )

    for column, info in numerical.items():

        minimum = info.get(
            "min"
        )

        if minimum is None:
            continue

        minimum = float(
            minimum
        )

        column_lower = column.lower()

        # -----------------------------------------------------
        # Percentage-like columns
        # -----------------------------------------------------

        if any(
            keyword in column_lower
            for keyword in percentage_keywords
        ):

            maximum = info.get(
                "max"
            )

            if (
                minimum < 0
                or (
                    maximum is not None
                    and float(maximum) > 100
                )
            ):

                findings.append({
                    "type": "invalid_values",
                    "severity": "high",
                    "message": (
                        f"Percentage-like column "
                        f"'{column}' contains "
                        f"values outside 0-100."
                    ),
                    "column": column,
                    "value": (
                        f"{minimum} to "
                        f"{info.get('max')}"
                    ),
                })

                continue

        # -----------------------------------------------------
        # Probability-like columns
        # -----------------------------------------------------

        if any(
            keyword in column_lower
            for keyword in probability_keywords
        ):

            maximum = info.get(
                "max"
            )

            if (
                minimum < 0
                or (
                    maximum is not None
                    and float(maximum) > 1
                )
            ):

                findings.append({
                    "type": "invalid_values",
                    "severity": "high",
                    "message": (
                        f"Probability-like column "
                        f"'{column}' contains "
                        f"values outside 0-1."
                    ),
                    "column": column,
                    "value": (
                        f"{minimum} to "
                        f"{info.get('max')}"
                    ),
                })

                continue

        # -----------------------------------------------------
        # Age-like columns
        # -----------------------------------------------------

        if (
            column_lower == "age"
            or column_lower.endswith("_age")
            or column_lower.startswith("age_")
        ):

            maximum = info.get(
                "max"
            )

            if (
                minimum < 0
                or (
                    maximum is not None
                    and float(maximum) > 120
                )
            ):

                findings.append({
                    "type": "invalid_values",
                    "severity": "high",
                    "message": (
                        f"Age-like column "
                        f"'{column}' contains "
                        f"values outside 0-120."
                    ),
                    "column": column,
                    "value": (
                        f"{minimum} to "
                        f"{info.get('max')}"
                    ),
                })

                continue

        # -----------------------------------------------------
        # Explicitly non-negative quantities
        # -----------------------------------------------------

        if any(
            keyword == column_lower
            or column_lower.startswith(
                f"{keyword}_"
            )
            or column_lower.endswith(
                f"_{keyword}"
            )
            for keyword in non_negative_keywords
        ):

            if minimum < 0:

                findings.append({
                    "type": "invalid_values",
                    "severity": "high",
                    "message": (
                        f"Column '{column}' "
                        f"contains negative values "
                        f"despite being a non-negative "
                        f"quantity."
                    ),
                    "column": column,
                    "value": (
                        f"{minimum} to "
                        f"{info.get('max')}"
                    ),
                })

    return findings


# =============================================================
# DATA QUALITY SCORE
# =============================================================

def calculate_data_quality_score(
    results: dict,
) -> int:

    score = 100

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing = results.get(
        "missing",
        {},
    )

    for info in missing.get(
        "columns",
        {},
    ).values():

        percentage = float(
            info.get(
                "percentage",
                0,
            )
        )

        if percentage > 50:
            score -= 15

        elif percentage >= 20:
            score -= 10

        elif percentage >= 5:
            score -= 5

    # ---------------------------------------------------------
    # Duplicates
    # ---------------------------------------------------------

    duplicates = results.get(
        "duplicates",
        {},
    )

    duplicate_percentage = float(
        duplicates.get(
            "duplicate_percentage",
            0,
        )
    )

    if duplicate_percentage > 10:
        score -= 10

    elif duplicate_percentage > 1:
        score -= 5

    # ---------------------------------------------------------
    # Target imbalance
    # ---------------------------------------------------------

    target = results.get(
        "target",
        {},
    )

    imbalance_ratio = target.get(
        "imbalance_ratio"
    )

    if imbalance_ratio is not None:

        imbalance_ratio = float(
            imbalance_ratio
        )

        if imbalance_ratio > 10:
            score -= 15

        elif imbalance_ratio > 5:
            score -= 10

        elif imbalance_ratio > 2:
            score -= 5

    # ---------------------------------------------------------
    # Constant columns
    # ---------------------------------------------------------

    cardinality = results.get(
        "cardinality",
        {},
    )

    for info in cardinality.values():

        unique_count = int(
            info.get(
                "unique_count",
                0,
            )
        )

        if unique_count == 1:
            score -= 5

    # ---------------------------------------------------------
    # Invalid / suspicious values
    # ---------------------------------------------------------

    findings = _get_quality_findings(
        results
    )

    invalid_count = sum(
        1
        for finding in findings
        if finding["type"]
        == "invalid_values"
    )

    score -= (
        invalid_count
        * 5
    )

    return max(
        0,
        min(
            100,
            score,
        ),
    )


# =============================================================
# STATISTICAL SIGNALS
# =============================================================

def get_statistical_signals(
    results: dict,
) -> list[dict]:

    signals = []

    # ---------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------

    correlations = results.get(
        "correlations",
        {},
    )

    for correlation in correlations.get(
        "numeric",
        [],
    ):

        value = float(
            correlation["pearson"]
        )

        absolute_value = abs(
            value
        )

        if absolute_value > 0.9:
            severity = "high"

        elif absolute_value > 0.7:
            severity = "medium"

        else:
            continue

        columns = [
            correlation["column_1"],
            correlation["column_2"],
        ]

        signals.append({
            "type": "correlation",
            "severity": severity,
            "message": (
                f"Strong correlation detected "
                f"between {columns[0]} and "
                f"{columns[1]}."
            ),
            "columns": columns,
            "value": round(
                value,
                4,
            ),
        })

    # ---------------------------------------------------------
    # Outliers
    # ---------------------------------------------------------

    outliers = results.get(
        "outliers",
        {},
    )

    for column, info in outliers.items():

        percentage = float(
            info.get(
                "outlier_percentage",
                0,
            )
        )

        if percentage > 10:
            severity = "high"

        elif percentage > 5:
            severity = "medium"

        elif percentage > 1:
            severity = "low"

        else:
            continue

        signals.append({
            "type": "outliers",
            "severity": severity,
            "message": (
                f"Column '{column}' "
                f"contains {percentage}% "
                f"outliers."
            ),
            "column": column,
            "value": percentage,
        })

    # ---------------------------------------------------------
    # Skewness
    # ---------------------------------------------------------

    numerical = results.get(
        "numerical",
        {},
    )

    for column, info in numerical.items():

        skewness = info.get(
            "skewness"
        )

        if skewness is None:
            continue

        skewness = float(
            skewness
        )

        absolute_skewness = abs(
            skewness
        )

        if absolute_skewness > 2:
            severity = "high"

        elif absolute_skewness > 1:
            severity = "medium"

        else:
            continue

        signals.append({
            "type": "skewness",
            "severity": severity,
            "message": (
                f"Column '{column}' "
                f"has notable skewness."
            ),
            "column": column,
            "value": round(
                skewness,
                4,
            ),
        })

    # ---------------------------------------------------------
    # High cardinality
    # ---------------------------------------------------------

    cardinality = results.get(
        "cardinality",
        {},
    )

    for column, info in cardinality.items():

        # Identifier columns are expected to have
        # high cardinality.
        if info.get(
            "is_id_like",
            False,
        ):
            continue

        # Only categorical columns should generate
        # high-cardinality signals.
        if info.get(
            "detected_type"
        ) != "categorical":
            continue

        unique_percentage = float(
            info.get(
                "unique_percentage",
                0,
            )
        )

        unique_count = int(
            info.get(
                "unique_count",
                0,
            )
        )

        if (
            unique_percentage >= 90
            and unique_count > 10
        ):

            signals.append({
                "type": "cardinality",
                "severity": "medium",
                "message": (
                    f"Column '{column}' "
                    f"has very high cardinality."
                ),
                "column": column,
                "value": unique_percentage,
            })

    return signals


# =============================================================
# HEALTH SCORE
# =============================================================

def calculate_health_score(
    results: dict,
) -> dict:

    score = calculate_data_quality_score(
        results
    )

    quality_findings = _get_quality_findings(
        results
    )

    statistical_signals = get_statistical_signals(
        results
    )

    # ---------------------------------------------------------
    # Grade
    # ---------------------------------------------------------

    if score >= 90:
        grade = "Excellent"

    elif score >= 75:
        grade = "Good"

    elif score >= 60:
        grade = "Fair"

    elif score >= 40:
        grade = "Poor"

    else:
        grade = "Critical"

    return {
        "score": score,
        "grade": grade,

        "data_quality": {
            "issues": quality_findings,
            "issue_count": len(
                quality_findings
            ),
        },

        "statistical_signals": {
            "signals": statistical_signals,
            "signal_count": len(
                statistical_signals
            ),
        },

        # Backward-compatible fields
        "quality_issues": quality_findings,
        "issues": quality_findings,
        "issue_count": len(
            quality_findings
        ),

        "finding_count": (
            len(quality_findings)
            + len(statistical_signals)
        ),
    }