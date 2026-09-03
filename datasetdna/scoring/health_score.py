from typing import Any


def calculate_health_score(
    results: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate an interpretable DatasetDNA health score.

    Profiler modules provide raw facts. This module is responsible
    for interpreting those facts, assigning severity, and applying
    score penalties.
    """

    score = 100
    issues: list[dict[str, Any]] = []

    def add_issue(
        issue_type: str,
        severity: str,
        penalty: int,
        message: str,
        **details: Any,
    ) -> None:
        nonlocal score

        score -= penalty

        issues.append(
            {
                "type": issue_type,
                "severity": severity,
                "penalty": penalty,
                "message": message,
                **details,
            }
        )

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing = results.get("missing", {})

    for column, details in missing.get("columns", {}).items():
        percentage = float(details.get("percentage", 0))

        if percentage > 50:
            add_issue(
                "missing",
                "severe",
                15,
                f"Column '{column}' has extremely high missingness.",
                column=column,
                value=percentage,
            )
        elif percentage > 20:
            add_issue(
                "missing",
                "high",
                10,
                f"Column '{column}' has high missingness.",
                column=column,
                value=percentage,
            )
        elif percentage > 5:
            add_issue(
                "missing",
                "medium",
                5,
                f"Column '{column}' has noticeable missingness.",
                column=column,
                value=percentage,
            )

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    duplicates = results.get("duplicates", {})
    duplicate_percentage = float(
        duplicates.get("duplicate_percentage", 0)
    )

    if duplicate_percentage > 10:
        add_issue(
            "duplicates",
            "high",
            10,
            "Dataset contains a high proportion of duplicate rows.",
            value=duplicate_percentage,
        )
    elif duplicate_percentage > 1:
        add_issue(
            "duplicates",
            "medium",
            5,
            "Dataset contains duplicate rows.",
            value=duplicate_percentage,
        )

    # ---------------------------------------------------------
    # Outliers
    # ---------------------------------------------------------

    outliers = results.get("outliers", {})

    for column, details in outliers.items():
        percentage = float(
            details.get("outlier_percentage", 0)
        )

        if percentage > 10:
            add_issue(
                "outliers",
                "high",
                8,
                f"Column '{column}' contains a high proportion of outliers.",
                column=column,
                value=percentage,
            )
        elif percentage > 5:
            add_issue(
                "outliers",
                "medium",
                4,
                f"Column '{column}' contains a noticeable proportion of outliers.",
                column=column,
                value=percentage,
            )
        elif percentage > 1:
            add_issue(
                "outliers",
                "low",
                2,
                f"Column '{column}' contains some potential outliers.",
                column=column,
                value=percentage,
            )

    # ---------------------------------------------------------
    # Numerical correlations
    # ---------------------------------------------------------

    correlations = results.get("correlations", {})

    for pair in correlations.get("numeric", []):
        correlation = abs(float(pair.get("pearson", 0)))

        if correlation > 0.90:
            add_issue(
                "correlation",
                "high",
                5,
                (
                    f"Very strong correlation detected between "
                    f"'{pair['column_1']}' and '{pair['column_2']}'."
                ),
                columns=[
                    pair["column_1"],
                    pair["column_2"],
                ],
                value=correlation,
            )
        elif correlation > 0.70:
            add_issue(
                "correlation",
                "medium",
                2,
                (
                    f"Strong correlation detected between "
                    f"'{pair['column_1']}' and '{pair['column_2']}'."
                ),
                columns=[
                    pair["column_1"],
                    pair["column_2"],
                ],
                value=correlation,
            )

    # ---------------------------------------------------------
    # Target imbalance
    # ---------------------------------------------------------

    target = results.get("target", {})

    if target.get("provided") and "error" not in target:
        imbalance_ratio = target.get("imbalance_ratio")

        if imbalance_ratio is not None:
            imbalance_ratio = float(imbalance_ratio)

            if imbalance_ratio > 10:
                add_issue(
                    "target_imbalance",
                    "severe",
                    15,
                    "Target classes are severely imbalanced.",
                    value=imbalance_ratio,
                )
            elif imbalance_ratio > 5:
                add_issue(
                    "target_imbalance",
                    "high",
                    10,
                    "Target classes are highly imbalanced.",
                    value=imbalance_ratio,
                )
            elif imbalance_ratio > 2:
                add_issue(
                    "target_imbalance",
                    "medium",
                    5,
                    "Target classes show noticeable imbalance.",
                    value=imbalance_ratio,
                )

    # ---------------------------------------------------------
    # Keep score inside valid range
    # ---------------------------------------------------------

    score = max(0, min(100, score))

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
        "issues": issues,
        "issue_count": len(issues),
    }
