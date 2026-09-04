from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


# =============================================================
# HELPERS
# =============================================================

def _severity_display(
    severity: str,
) -> str:

    if severity == "severe":
        return "[bold red]🔴 SEVERE[/bold red]"

    if severity == "high":
        return "[red]🔴 HIGH[/red]"

    if severity == "medium":
        return "[yellow]🟡 MEDIUM[/yellow]"

    return "[dim]⚪ LOW[/dim]"


def _score_style(
    score: int,
) -> tuple[str, str]:

    if score >= 90:
        return "bold green", "🟢"

    if score >= 75:
        return "bold yellow", "🟡"

    if score >= 60:
        return "bold yellow", "🟠"

    return "bold red", "🔴"


def _recommendation_display(
    severity: str,
) -> str:

    if severity == "high":
        return "[red]🔴 HIGH[/red]"

    if severity == "medium":
        return "[yellow]🟡 MEDIUM[/yellow]"

    return "[dim]⚪ LOW[/dim]"


# =============================================================
# HEADER
# =============================================================

def render_header() -> None:

    console.print()

    console.print(
        Panel.fit(
            "[bold cyan]🧬 DatasetDNA[/bold cyan]\n"
            "[bold white]Automated Dataset Health Report[/bold white]\n"
            "[dim]Profile • Detect • Score • Recommend[/dim]",
            border_style="cyan",
        )
    )


# =============================================================
# HEALTH SCORE
# =============================================================

def render_health_score(
    health: dict[str, Any],
) -> None:

    score = int(
        health["score"]
    )

    grade = health["grade"]

    quality_count = health[
        "data_quality"
    ]["issue_count"]

    signal_count = health[
        "statistical_signals"
    ]["signal_count"]

    score_style, status_icon = _score_style(
        score
    )

    content = (
        f"{status_icon} "
        "[bold white]Dataset Health[/bold white]\n\n"
        f"[{score_style}]"
        f"{score} / 100"
        f"[/{score_style}]\n"
        f"[bold white]{grade}[/bold white]\n\n"
        "[dim]"
        f"{quality_count} quality issue(s)  •  "
        f"{signal_count} statistical signal(s)"
        "[/dim]"
    )

    console.print()

    console.print(
        Panel(
            content,
            title="🧬 HEALTH SCORE",
            border_style="cyan",
            expand=False,
        )
    )


# =============================================================
# DATA QUALITY
# =============================================================

def render_data_quality(
    health: dict[str, Any],
) -> None:

    issues = health[
        "data_quality"
    ]["issues"]

    console.print()

    if not issues:

        console.print(
            Panel(
                "[bold green]"
                "✓ No significant data-quality "
                "issues detected."
                "[/bold green]",
                title="🧬 DATA QUALITY",
                border_style="green",
            )
        )

        return

    table = Table(
        title="🧬 DATA QUALITY",
        show_header=True,
        header_style="bold red",
        expand=False,
    )

    table.add_column(
        "Severity",
        no_wrap=True,
    )

    table.add_column(
        "Issue",
    )

    table.add_column(
        "Details",
    )

    for issue in issues:

        table.add_row(
            _severity_display(
                issue.get(
                    "severity",
                    "low",
                )
            ),
            issue["message"],
            _format_quality_details(
                issue
            ),
        )

    console.print(table)


def _format_quality_details(
    issue: dict[str, Any],
) -> str:

    issue_type = issue.get(
        "type"
    )

    if issue_type == "missing":

        return (
            f"Column: {issue['column']} | "
            f"Missing: {issue['value']}%"
        )

    if issue_type == "duplicates":

        return (
            f"Duplicate rows: "
            f"{issue['value']}%"
        )

    if issue_type == "target_imbalance":

        return (
            f"Imbalance ratio: "
            f"{issue['value']}:1"
        )

    if issue_type == "constant":

        return (
            f"Column: {issue['column']} | "
            f"Unique values: {issue['value']}"
        )

    if issue_type == "invalid_values":

        return (
            f"Observed range: "
            f"{issue.get('value')}"
        )

    return ""


# =============================================================
# STATISTICAL SIGNALS
# =============================================================

def render_statistical_signals(
    health: dict[str, Any],
) -> None:

    signals = health[
        "statistical_signals"
    ]["signals"]

    console.print()

    if not signals:

        console.print(
            Panel(
                "[bold green]"
                "✓ No notable statistical "
                "signals detected."
                "[/bold green]",
                title="📊 STATISTICAL SIGNALS",
                border_style="green",
            )
        )

        return

    table = Table(
        title="📊 STATISTICAL SIGNALS",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column(
        "Level",
        no_wrap=True,
    )

    table.add_column(
        "Signal",
    )

    table.add_column(
        "Details",
    )

    for signal in signals:

        table.add_row(
            _signal_display(
                signal.get(
                    "severity",
                    "low",
                )
            ),
            signal["message"],
            _format_signal_details(
                signal
            ),
        )

    console.print(table)


def _signal_display(
    severity: str,
) -> str:

    if severity == "high":
        return "[red]🔎 HIGH[/red]"

    if severity == "medium":
        return "[yellow]🔎 MEDIUM[/yellow]"

    return "[dim]🔎 LOW[/dim]"


def _format_signal_details(
    signal: dict[str, Any],
) -> str:

    signal_type = signal.get(
        "type"
    )

    if signal_type == "correlation":

        columns = signal["columns"]

        return (
            f"{columns[0]} ↔ "
            f"{columns[1]} | "
            f"Pearson: {signal['value']}"
        )

    if signal_type == "outliers":

        return (
            f"Column: {signal['column']} | "
            f"Outliers: {signal['value']}%"
        )

    if signal_type == "skewness":

        return (
            f"Column: {signal['column']} | "
            f"Skewness: {signal['value']}"
        )

    if signal_type == "cardinality":

        return (
            f"Column: {signal['column']} | "
            f"Unique: {signal['value']}%"
        )

    return ""


# =============================================================
# CATEGORY CONSISTENCY
# =============================================================

def render_category_consistency(
    category_consistency: dict[str, Any],
) -> None:
    """
    Display categorical columns containing multiple
    representations of the same category.

    Example:

        Male / male / M
        Female / female / F
    """

    console.print()

    if not category_consistency:

        console.print(
            Panel(
                "[bold green]"
                "✓ No categorical representation "
                "inconsistencies detected."
                "[/bold green]",
                title="🏷 CATEGORY CONSISTENCY",
                border_style="green",
            )
        )

        return

    table = Table(
        title="🏷 CATEGORY CONSISTENCY",
        show_header=True,
        header_style="bold yellow",
        expand=False,
    )

    table.add_column(
        "Column",
        no_wrap=True,
    )

    table.add_column(
        "Canonical Category",
        no_wrap=True,
    )

    table.add_column(
        "Representations",
    )

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

        for canonical, values in groups.items():

            if not isinstance(
                values,
                (list, tuple, set),
            ):
                continue

            table.add_row(
                str(column),
                str(canonical),
                ", ".join(
                    str(value)
                    for value in values
                ),
            )

    console.print(table)

    console.print(
        Panel(
            "[yellow]⚠[/yellow] "
            "Multiple representations of the same "
            "category were detected. Standardize "
            "categorical values before model training.",
            border_style="yellow",
        )
    )


# =============================================================
# RECOMMENDATIONS
# =============================================================

def render_recommendations(
    recommendations: list[dict[str, Any]],
) -> None:

    console.print()

    if not recommendations:

        console.print(
            Panel(
                "[bold green]"
                "✓ No specific recommendations. "
                "Dataset looks ready for further analysis."
                "[/bold green]",
                title="💡 RECOMMENDATIONS",
                border_style="green",
            )
        )

        return

    table = Table(
        title="💡 RECOMMENDATIONS",
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )

    table.add_column(
        "Priority",
        no_wrap=True,
    )

    table.add_column(
        "Action",
    )

    for recommendation in recommendations:

        table.add_row(
            _recommendation_display(
                recommendation.get(
                    "severity",
                    "low",
                )
            ),
            recommendation.get(
                "message",
                "",
            ),
        )

    console.print(table)


# =============================================================
# OVERVIEW
# =============================================================

def render_overview(
    overview: dict[str, Any],
    missing: dict[str, Any],
    duplicates: dict[str, Any],
) -> None:

    table = Table(
        title="📋 DATASET OVERVIEW",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Metric")
    table.add_column("Value")

    table.add_row(
        "Rows",
        str(overview["rows"]),
    )

    table.add_row(
        "Columns",
        str(overview["columns"]),
    )

    table.add_row(
        "Memory",
        f'{overview["memory_usage_mb"]} MB',
    )

    table.add_row(
        "Missing Cells",
        str(
            missing[
                "total_missing_cells"
            ]
        ),
    )

    table.add_row(
        "Duplicate Rows",
        str(
            duplicates[
                "duplicate_count"
            ]
        ),
    )

    table.add_row(
        "Duplicate %",
        f'{duplicates["duplicate_percentage"]}%',
    )

    console.print()
    console.print(table)


# =============================================================
# SCHEMA
# =============================================================

def render_schema(
    schema: dict[str, Any],
) -> None:

    table = Table(
        title="🔧 SCHEMA",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Column")
    table.add_column("Data Type")

    for column, dtype in schema.items():

        table.add_row(
            column,
            dtype,
        )

    console.print()
    console.print(table)


# =============================================================
# MISSING
# =============================================================

def render_missing(
    missing: dict[str, Any],
) -> None:

    table = Table(
        title="⚠ MISSING VALUES",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Column")
    table.add_column("Missing Count")
    table.add_column("Missing %")

    for column, result in missing[
        "columns"
    ].items():

        table.add_row(
            column,
            str(result["count"]),
            f'{result["percentage"]}%',
        )

    console.print()
    console.print(table)


# =============================================================
# CARDINALITY
# =============================================================

def render_cardinality(
    cardinality: dict[str, Any],
) -> None:

    table = Table(
        title="🔢 CARDINALITY",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Unique Values")
    table.add_column("Unique %")

    for column, result in cardinality.items():

        detected_type = result.get(
            "detected_type",
            "unknown",
        )

        table.add_row(
            column,
            detected_type,
            str(
                result["unique_count"]
            ),
            f'{result["unique_percentage"]}%',
        )

    console.print()
    console.print(table)


# =============================================================
# NUMERICAL
# =============================================================

def render_numerical(
    numerical: dict[str, Any],
) -> None:

    console.print()

    if not numerical:

        console.print(
            Panel(
                "No numerical columns detected.",
                title="Numerical Analysis",
            )
        )

        return

    table = Table(
        title="📈 NUMERICAL ANALYSIS",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Column")
    table.add_column("Mean")
    table.add_column("Median")
    table.add_column("Std")
    table.add_column("Min")
    table.add_column("Max")
    table.add_column("Skew")

    for column, result in numerical.items():

        table.add_row(
            column,
            str(result["mean"]),
            str(result["median"]),
            str(result["std"]),
            str(result["min"]),
            str(result["max"]),
            str(result["skewness"]),
        )

    console.print(table)


# =============================================================
# CATEGORICAL
# =============================================================

def render_categorical(
    categorical: dict[str, Any],
) -> None:

    console.print()

    if not categorical:

        console.print(
            Panel(
                "No categorical columns detected.",
                title="Categorical Analysis",
            )
        )

        return

    table = Table(
        title="🏷 CATEGORICAL ANALYSIS",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Column")
    table.add_column("Unique Values")
    table.add_column("Top Categories")

    for column, result in categorical.items():

        categories = result[
            "categories"
        ]

        sorted_categories = sorted(
            categories.items(),
            key=lambda item: item[1]["count"],
            reverse=True,
        )

        top_categories = sorted_categories[:5]

        category_text = "\n".join(
            f"{category}: "
            f"{details['count']} "
            f"({details['percentage']}%)"
            for category, details
            in top_categories
        )

        table.add_row(
            column,
            str(
                result["unique_count"]
            ),
            (
                category_text
                if category_text
                else "No values"
            ),
        )

    console.print(table)


# =============================================================
# OUTLIERS
# =============================================================

def render_outliers(
    outliers: dict[str, Any],
) -> None:

    console.print()

    if not outliers:

        console.print(
            Panel(
                "No numerical columns available "
                "for outlier analysis.",
                title="Outlier Analysis",
            )
        )

        return

    table = Table(
        title="📌 OUTLIER ANALYSIS",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Column")
    table.add_column("Outliers")
    table.add_column("Outlier %")
    table.add_column("Lower Bound")
    table.add_column("Upper Bound")

    for column, result in outliers.items():

        table.add_row(
            column,
            str(
                result["outlier_count"]
            ),
            f'{result["outlier_percentage"]}%',
            str(
                result["lower_bound"]
            ),
            str(
                result["upper_bound"]
            ),
        )

    console.print(table)


# =============================================================
# CORRELATIONS
# =============================================================

def render_correlations(
    correlations: dict[str, Any],
) -> None:

    numeric_correlations = correlations.get(
        "numeric",
        [],
    )

    console.print()

    if numeric_correlations:

        table = Table(
            title="🔗 NUMERICAL CORRELATIONS",
            show_header=True,
            header_style="bold cyan",
            expand=False,
        )

        table.add_column("Column 1")
        table.add_column("Column 2")
        table.add_column("Pearson")

        for result in numeric_correlations:

            table.add_row(
                result["column_1"],
                result["column_2"],
                str(result["pearson"]),
            )

        console.print(table)

    else:

        console.print(
            Panel(
                "No numerical column pairs available.",
                title="Numerical Correlations",
            )
        )

    categorical_correlations = correlations.get(
        "categorical",
        [],
    )

    if categorical_correlations:

        console.print()

        table = Table(
            title="🔗 CATEGORICAL ASSOCIATIONS",
            show_header=True,
            header_style="bold cyan",
            expand=False,
        )

        table.add_column("Column 1")
        table.add_column("Column 2")
        table.add_column("Cramer's V")

        for result in categorical_correlations:

            table.add_row(
                result["column_1"],
                result["column_2"],
                str(result["cramers_v"]),
            )

        console.print(table)


# =============================================================
# TARGET
# =============================================================

def render_target(
    target: dict[str, Any],
) -> None:

    table = Table(
        title="🎯 TARGET ANALYSIS",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )

    table.add_column("Metric")
    table.add_column("Value")

    if not target["provided"]:

        table.add_row(
            "Target",
            "Not provided",
        )

    elif "error" in target:

        table.add_row(
            "Target",
            target["column"],
        )

        table.add_row(
            "Status",
            target["error"],
        )

    elif target.get(
        "type"
    ) == "empty":

        table.add_row(
            "Target",
            target["column"],
        )

        table.add_row(
            "Status",
            "Target contains no non-null values.",
        )

    else:

        table.add_row(
            "Target",
            target["column"],
        )

        table.add_row(
            "Type",
            target["type"],
        )

        table.add_row(
            "Classes",
            str(
                target["class_count"]
            ),
        )

        imbalance_ratio = target[
            "imbalance_ratio"
        ]

        if imbalance_ratio is not None:

            table.add_row(
                "Imbalance Ratio",
                f"{imbalance_ratio}:1",
            )

        else:

            table.add_row(
                "Imbalance Ratio",
                "N/A",
            )

        for category, details in target[
            "class_distribution"
        ].items():

            table.add_row(
                f"Class: {category}",
                f'{details["count"]} '
                f'({details["percentage"]}%)',
            )

    console.print()
    console.print(table)


# =============================================================
# COMPLETION
# =============================================================

def render_completion(
    recommendation_count: int,
) -> None:

    console.print()

    console.print(
        Panel(
            "[bold green]"
            "✓ DatasetDNA analysis completed successfully."
            "[/bold green]\n\n"
            f"[dim]"
            f"{recommendation_count} actionable "
            "recommendation(s) generated."
            f"[/dim]",
            border_style="green",
        )
    )

    console.print()


# =============================================================
# COMPLETE REPORT
# =============================================================

def render_report(
    results: dict[str, Any],
    health: dict[str, Any],
    recommendations: list[dict[str, Any]] | None = None,
) -> None:

    if recommendations is None:
        recommendations = []

    render_header()

    render_health_score(
        health
    )

    render_data_quality(
        health
    )

    render_statistical_signals(
        health
    )

    render_recommendations(
        recommendations
    )

    render_overview(
        results["overview"],
        results["missing"],
        results["duplicates"],
    )

    render_schema(
        results["schema"]
    )

    render_missing(
        results["missing"]
    )

    render_cardinality(
        results["cardinality"]
    )

    render_numerical(
        results["numerical"]
    )

    render_categorical(
        results["categorical"]
    )

    render_category_consistency(
        results.get(
            "category_consistency",
            {},
        )
    )

    render_outliers(
        results["outliers"]
    )

    render_correlations(
        results["correlations"]
    )

    render_target(
        results["target"]
    )

    render_completion(
        len(recommendations)
    )