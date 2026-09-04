from __future__ import annotations

import html

import plotly.graph_objects as go
from plotly.offline import plot


# ============================================================
# HELPERS
# ============================================================

def _escape(value) -> str:
    return html.escape(str(value))


def _plot_to_html(figure) -> str:
    return plot(
        figure,
        output_type="div",
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


def _score_class(score: int | float) -> str:
    if score >= 80:
        return "good"

    if score >= 50:
        return "warning"

    return "danger"


def _severity_class(severity: str) -> str:
    severity = str(severity).lower()

    if severity == "high":
        return "severity-high"

    if severity == "medium":
        return "severity-medium"

    return "severity-low"


# ============================================================
# HEALTH GAUGE
# ============================================================

def _build_health_gauge(
    score: int | float,
) -> str:

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "font": {
                    "size": 54,
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "dtick": 20,
                },
                "bar": {
                    "thickness": 0.75,
                },
                "steps": [
                    {
                        "range": [0, 50],
                    },
                    {
                        "range": [50, 80],
                    },
                    {
                        "range": [80, 100],
                    },
                ],
                "threshold": {
                    "line": {
                        "width": 4,
                    },
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )

    figure.update_layout(
        height=300,
        margin=dict(
            l=35,
            r=35,
            t=25,
            b=10,
        ),
        template="plotly_white",
    )

    return _plot_to_html(figure)


# ============================================================
# MISSING VALUES
# ============================================================

def _build_missing_chart(
    results: dict,
) -> str:

    missing = results.get(
        "missing",
        {},
    )

    columns = missing.get(
        "columns",
        {},
    )

    filtered = {
        column: data
        for column, data in columns.items()
        if data.get("percentage", 0) > 0
    }

    if not filtered:
        return (
            '<p class="empty">'
            "No missing values detected."
            "</p>"
        )

    filtered = dict(
        sorted(
            filtered.items(),
            key=lambda item: item[1].get(
                "percentage",
                0,
            ),
            reverse=True,
        )
    )

    names = list(filtered.keys())

    percentages = [
        data.get("percentage", 0)
        for data in filtered.values()
    ]

    counts = [
        data.get("count", 0)
        for data in filtered.values()
    ]

    figure = go.Figure(
        go.Bar(
            y=names,
            x=percentages,
            orientation="h",
            text=[
                f"{value:.1f}%"
                for value in percentages
            ],
            textposition="auto",
            customdata=counts,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Missing: %{x:.2f}%<br>"
                "Count: %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="Missing Values by Column",
        xaxis_title="Missing %",
        yaxis_title="",
        template="plotly_white",
        height=max(
            300,
            len(names) * 65,
        ),
        margin=dict(
            l=140,
            r=40,
            t=65,
            b=60,
        ),
        showlegend=False,
    )

    return _plot_to_html(figure)


# ============================================================
# NUMERICAL ANALYSIS
# ============================================================

def _build_numerical_chart(
    results: dict,
) -> str:

    numerical = results.get(
        "numerical",
        {},
    )

    if not numerical:
        return (
            '<p class="empty">'
            "No numerical data available."
            "</p>"
        )

    columns = list(numerical.keys())

    means = [
        data.get("mean")
        for data in numerical.values()
    ]

    medians = [
        data.get("median")
        for data in numerical.values()
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            name="Mean",
            x=columns,
            y=means,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean: %{y}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Bar(
            name="Median",
            x=columns,
            y=medians,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Median: %{y}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="Mean vs Median",
        xaxis_title="Column",
        yaxis_title="Value",
        barmode="group",
        template="plotly_white",
        height=420,
        margin=dict(
            l=55,
            r=30,
            t=65,
            b=100,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return _plot_to_html(figure)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def _build_outlier_chart(
    results: dict,
) -> str:

    outliers = results.get(
        "outliers",
        {},
    )

    filtered = {
        column: data
        for column, data in outliers.items()
        if data.get("outlier_count", 0) > 0
    }

    if not filtered:
        return (
            '<p class="empty">'
            "No outliers detected."
            "</p>"
        )

    filtered = dict(
        sorted(
            filtered.items(),
            key=lambda item: item[1].get(
                "outlier_percentage",
                0,
            ),
            reverse=True,
        )
    )

    columns = list(filtered.keys())

    percentages = [
        data.get("outlier_percentage", 0)
        for data in filtered.values()
    ]

    counts = [
        data.get("outlier_count", 0)
        for data in filtered.values()
    ]

    figure = go.Figure(
        go.Bar(
            x=columns,
            y=percentages,
            text=[
                f"{value:.1f}%"
                for value in percentages
            ],
            textposition="auto",
            customdata=counts,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Outliers: %{customdata}<br>"
                "Percentage: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="Outlier Percentage",
        xaxis_title="Column",
        yaxis_title="Outlier %",
        template="plotly_white",
        height=420,
        margin=dict(
            l=50,
            r=30,
            t=65,
            b=100,
        ),
        showlegend=False,
    )

    return _plot_to_html(figure)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def _build_correlation_chart(
    results: dict,
) -> str:

    correlations = results.get(
        "correlations",
        {},
    )

    numeric = correlations.get(
        "numeric",
        [],
    )

    if not numeric:
        return (
            '<p class="empty">'
            "No numerical correlations available."
            "</p>"
        )

    columns = set()

    for item in numeric:
        columns.add(
            item["column_1"]
        )
        columns.add(
            item["column_2"]
        )

    columns = sorted(columns)

    matrix = [
        [0.0 for _ in columns]
        for _ in columns
    ]

    index = {
        column: position
        for position, column in enumerate(columns)
    }

    for item in numeric:

        column_1 = item["column_1"]
        column_2 = item["column_2"]
        value = item["pearson"]

        i = index[column_1]
        j = index[column_2]

        matrix[i][j] = value
        matrix[j][i] = value

    for i in range(len(columns)):
        matrix[i][i] = 1.0

    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=columns,
            y=columns,
            zmin=-1,
            zmax=1,
            text=[
                [
                    f"{value:.2f}"
                    for value in row
                ]
                for row in matrix
            ],
            texttemplate="%{text}",
            hovertemplate=(
                "<b>%{y}</b> × "
                "<b>%{x}</b><br>"
                "Correlation: %{z:.4f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="Numerical Correlation Heatmap",
        template="plotly_white",
        height=520,
        margin=dict(
            l=60,
            r=30,
            t=65,
            b=80,
        ),
    )

    return _plot_to_html(figure)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def _build_target_chart(
    results: dict,
) -> str:

    target = results.get(
        "target",
        {},
    )

    if not target.get("provided"):
        return (
            '<p class="empty">'
            "No target column provided."
            "</p>"
        )

    # Regression targets are continuous.
    # Class distribution / class imbalance does not apply.
    if target.get("task_type") == "regression":
        return (
            '<p class="empty">'
            "Target is continuous (regression) — "
            "class distribution not applicable."
            "</p>"
        )

    distribution = target.get(
        "class_distribution",
        {},
    )

    if not distribution:
        return (
            '<p class="empty">'
            "No target distribution available."
            "</p>"
        )

    labels = list(
        distribution.keys()
    )

    values = [
        item.get("count", 0)
        for item in distribution.values()
    ]

    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            textinfo="label+percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Count: %{value}<br>"
                "Percentage: %{percent}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=(
            "Target Distribution — "
            f"{target.get('column')}"
        ),
        template="plotly_white",
        height=420,
        margin=dict(
            l=30,
            r=30,
            t=65,
            b=30,
        ),
        showlegend=True,
    )

    return _plot_to_html(figure)


# ============================================================
# MAIN HTML REPORT
# ============================================================

def render_html_report(
    results: dict,
    output_path: str = "datasetdna_report.html",
) -> str:

    overview = results.get(
        "overview",
        {},
    )

    health = results.get(
        "health",
        {},
    )

    recommendations = results.get(
        "recommendations",
        [],
    )

    score = health.get(
        "score",
        0,
    )

    grade = health.get(
        "grade",
        "Unknown",
    )

    quality = health.get(
        "data_quality",
        {},
    )

    issues = quality.get(
        "issues",
        [],
    )

    schema = results.get(
        "schema",
        {},
    )

    missing = results.get(
        "missing",
        {},
    )

    cardinality = results.get(
        "cardinality",
        {},
    )

    category_consistency = results.get(
        "category_consistency",
        {},
    )

    numerical = results.get(
        "numerical",
        {},
    )

    target = results.get(
        "target",
        {},
    )

    correlations = results.get(
        "correlations",
        {},
    )

    rows = overview.get(
        "rows",
        0,
    )

    columns = overview.get(
        "columns",
        0,
    )

    memory = overview.get(
        "memory_usage_mb",
        0,
    )

    # ========================================================
    # COUNT STATISTICS
    # ========================================================

    missing_cells = missing.get(
        "total_missing_cells",
        0,
    )

    numeric_count = len(numerical)

    correlation_count = len(
        correlations.get(
            "numeric",
            [],
        )
    )

    outlier_count = sum(
        data.get(
            "outlier_count",
            0,
        )
        for data in results.get(
            "outliers",
            {},
        ).values()
    )

    # ========================================================
    # TABLE DATA
    # ========================================================

    schema_rows = ""

    for column, dtype in schema.items():

        schema_rows += f"""
        <tr>
            <td>
                <strong>
                    {_escape(column)}
                </strong>
            </td>

            <td>
                <span class="type-badge">
                    {_escape(dtype)}
                </span>
            </td>
        </tr>
        """

    missing_rows = ""

    for column, data in missing.get(
        "columns",
        {},
    ).items():

        percentage = data.get(
            "percentage",
            0,
        )

        missing_rows += f"""
        <tr>

            <td>
                {_escape(column)}
            </td>

            <td>
                {data.get("count", 0)}
            </td>

            <td>
                <div class="progress-row">

                    <div class="progress">
                        <div
                            class="progress-fill"
                            style="width: {percentage}%"
                        ></div>
                    </div>

                    <span>
                        {percentage}%
                    </span>

                </div>
            </td>

        </tr>
        """

    cardinality_rows = ""

    for column, data in cardinality.items():

        cardinality_rows += f"""
        <tr>

            <td>
                {_escape(column)}
            </td>

            <td>
                {data.get("unique_count", 0)}
            </td>

            <td>
                {data.get("unique_percentage", 0)}%
            </td>

            <td>
                <span class="type-badge">
                    {_escape(
                        data.get(
                            "detected_type",
                            "unknown",
                        )
                    )}
                </span>
            </td>

        </tr>
        """

    # ========================================================
    # CATEGORY CONSISTENCY
    # ========================================================

    category_consistency_rows = ""

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

            representations = ", ".join(
                _escape(value)
                for value in values
            )

            category_consistency_rows += f"""
            <tr>

                <td>
                    <strong>
                        {_escape(column)}
                    </strong>
                </td>

                <td>
                    <span class="type-badge">
                        {_escape(canonical)}
                    </span>
                </td>

                <td>
                    {representations}
                </td>

            </tr>
            """

    if category_consistency_rows:

        category_consistency_content = f"""
        <div class="table-wrapper">

            <table>

                <thead>

                    <tr>
                        <th>Column</th>
                        <th>Canonical Category</th>
                        <th>Representations</th>
                    </tr>

                </thead>

                <tbody>

                    {category_consistency_rows}

                </tbody>

            </table>

        </div>

        <div
            class="issue-card severity-medium"
            style="margin-top: 18px;"
        >

            <div class="issue-severity">
                MEDIUM
            </div>

            <div class="issue-message">
                Multiple representations of the same
                category were detected. Standardize
                categorical values before model training.
            </div>

        </div>
        """

    else:

        category_consistency_content = """
        <div class="empty-state">

            <div class="empty-icon">
                ✓
            </div>

            <strong>
                No categorical representation inconsistencies detected
            </strong>

            <span>
                Categorical values appear consistently represented.
            </span>

        </div>
        """

    # ========================================================
    # ISSUE CARDS
    # ========================================================

    issue_cards = ""

    for issue in issues:

        severity = issue.get(
            "severity",
            "info",
        )

        message = issue.get(
            "message",
            "Data quality issue",
        )

        issue_cards += f"""
        <div
            class="issue-card
            {_severity_class(severity)}"
        >

            <div class="issue-severity">
                {_escape(
                    str(severity).upper()
                )}
            </div>

            <div class="issue-message">
                {_escape(message)}
            </div>

        </div>
        """

    if not issue_cards:

        issue_cards = """
        <div class="empty-state">
            <div class="empty-icon">✓</div>
            <strong>No quality issues detected</strong>
            <span>Your dataset looks healthy.</span>
        </div>
        """

    # ========================================================
    # RECOMMENDATION CARDS
    # ========================================================

    recommendation_cards = ""

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):

        if isinstance(
            recommendation,
            dict,
        ):

            title = recommendation.get(
                "title",
                f"Recommendation {index}",
            )

            message = recommendation.get(
                "recommendation",
                recommendation.get(
                    "message",
                    str(recommendation),
                ),
            )

        else:

            title = (
                f"Recommendation {index}"
            )

            message = str(
                recommendation
            )

        recommendation_cards += f"""
        <div class="recommendation-card">

            <div class="recommendation-number">
                {index:02d}
            </div>

            <div>

                <div class="recommendation-title">
                    {_escape(title)}
                </div>

                <div class="recommendation-message">
                    {_escape(message)}
                </div>

            </div>

        </div>
        """

    if not recommendation_cards:

        recommendation_cards = """
        <div class="empty-state">
            <div class="empty-icon">✓</div>
            <strong>No recommendations</strong>
            <span>No immediate actions are required.</span>
        </div>
        """

    # ========================================================
    # CHARTS
    # ========================================================

    health_gauge = _build_health_gauge(
        score
    )

    missing_chart = _build_missing_chart(
        results
    )

    numerical_chart = _build_numerical_chart(
        results
    )

    outlier_chart = _build_outlier_chart(
        results
    )

    correlation_chart = _build_correlation_chart(
        results
    )

    target_chart = _build_target_chart(
        results
    )

    score_class = _score_class(
        score
    )

    # ========================================================
    # HTML DOCUMENT
    # ========================================================

    document = f"""
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>
        DatasetDNA — Dataset Health Report
    </title>

    <script
        src="https://cdn.plot.ly/plotly-2.35.2.min.js"
    ></script>

    <style>

        * {{
            box-sizing: border-box;
        }}

        :root {{
            --background: #f5f7fb;
            --surface: #ffffff;
            --text: #172033;
            --muted: #667085;
            --border: #eaecf0;
            --purple: #635bff;
            --green: #16803c;
            --orange: #b54708;
            --red: #b42318;
        }}

        html {{
            scroll-behavior: smooth;
        }}

        body {{
            margin: 0;

            background:
                linear-gradient(
                    180deg,
                    #f8f9fc 0%,
                    #f3f5fa 100%
                );

            color: var(--text);

            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            line-height: 1.5;
        }}

        .container {{
            max-width: 1240px;

            margin: auto;

            padding:
                45px 24px 70px;
        }}

        /* ====================================================
           HEADER
           ==================================================== */

        header {{
            display: flex;

            align-items: center;

            justify-content: space-between;

            margin-bottom: 32px;
        }}

        .brand {{
            display: flex;

            align-items: center;

            gap: 12px;
        }}

        .brand-icon {{
            font-size: 38px;
        }}

        .brand-name {{
            margin: 0;

            font-size: 36px;

            line-height: 1;

            letter-spacing: -1.5px;
        }}

        .subtitle {{
            color: var(--muted);

            margin-top: 9px;

            font-size: 15px;
        }}

        .badge {{
            padding:
                8px 14px;

            border-radius: 999px;

            background: #eef0ff;

            color: #5147d9;

            font-size: 13px;

            font-weight: 700;
        }}

        /* ====================================================
           CARDS
           ==================================================== */

        .card {{
            background: var(--surface);

            border:
                1px solid rgba(16, 24, 40, 0.04);

            border-radius: 20px;

            box-shadow:
                0 8px 30px
                rgba(16, 24, 40, 0.055);
        }}

        /* ====================================================
           HEALTH
           ==================================================== */

        .health-card {{
            padding: 24px;

            margin-bottom: 24px;
        }}

        .health-header {{
            display: flex;

            align-items: center;

            justify-content: space-between;

            margin-bottom: 5px;
        }}

        .health-header h2 {{
            margin: 0;

            font-size: 20px;
        }}

        .grade {{
            font-weight: 700;
        }}

        .good {{
            color: var(--green);
        }}

        .warning {{
            color: var(--orange);
        }}

        .danger {{
            color: var(--red);
        }}

        /* ====================================================
           METRICS
           ==================================================== */

        .metrics {{
            display: grid;

            grid-template-columns:
                repeat(
                    4,
                    minmax(0, 1fr)
                );

            gap: 18px;

            margin-bottom: 24px;
        }}

        .metric {{
            padding: 22px;
        }}

        .metric-label {{
            color: var(--muted);

            font-size: 13px;

            font-weight: 600;

            margin-bottom: 6px;
        }}

        .metric-value {{
            font-size: 30px;

            font-weight: 800;

            letter-spacing: -0.8px;
        }}

        .metric-detail {{
            color: var(--muted);

            font-size: 12px;

            margin-top: 4px;
        }}

        /* ====================================================
           SECTION
           ==================================================== */

        .section {{
            padding: 25px;

            margin-bottom: 24px;
        }}

        .section-header {{
            margin-bottom: 18px;
        }}

        .section-title {{
            margin: 0;

            font-size: 21px;

            letter-spacing: -0.3px;
        }}

        .section-description {{
            margin:
                5px 0 0;

            color: var(--muted);

            font-size: 14px;
        }}

        /* ====================================================
           CHART GRID
           ==================================================== */

        .chart-grid {{
            display: grid;

            grid-template-columns:
                repeat(
                    2,
                    minmax(0, 1fr)
                );

            gap: 24px;
        }}

        .chart-card {{
            padding: 24px;
        }}

        .chart-card .section-title {{
            font-size: 19px;
        }}

        /* ====================================================
           TABLES
           ==================================================== */

        .table-wrapper {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;

            border-collapse: collapse;
        }}

        th {{
            padding:
                13px 12px;

            text-align: left;

            color: var(--muted);

            font-size: 12px;

            text-transform: uppercase;

            letter-spacing: 0.4px;

            border-bottom:
                1px solid var(--border);
        }}

        td {{
            padding:
                14px 12px;

            border-bottom:
                1px solid var(--border);

            font-size: 14px;
        }}

        tbody tr:last-child td {{
            border-bottom: none;
        }}

        .type-badge {{
            display: inline-block;

            padding:
                4px 9px;

            border-radius: 7px;

            background: #f2f4f7;

            color: #475467;

            font-size: 12px;

            font-weight: 600;
        }}

        /* ====================================================
           PROGRESS
           ==================================================== */

        .progress-row {{
            display: flex;

            align-items: center;

            gap: 10px;
        }}

        .progress {{
            width: 130px;

            height: 7px;

            background: #eaecf0;

            border-radius: 999px;

            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;

            background: var(--purple);

            border-radius: inherit;
        }}

        /* ====================================================
           ISSUES
           ==================================================== */

        .issue-grid {{
            display: grid;

            grid-template-columns:
                repeat(
                    2,
                    minmax(0, 1fr)
                );

            gap: 14px;
        }}

        .issue-card {{
            padding: 17px;

            border-radius: 13px;

            border-left:
                4px solid;

            background: #fafafa;
        }}

        .severity-high {{
            border-color: var(--red);

            background: #fff7f7;
        }}

        .severity-medium {{
            border-color: var(--orange);

            background: #fffbf5;
        }}

        .severity-low {{
            border-color: var(--green);

            background: #f6fdf8;
        }}

        .issue-severity {{
            font-size: 11px;

            font-weight: 800;

            letter-spacing: 0.6px;

            margin-bottom: 5px;
        }}

        .issue-message {{
            font-size: 14px;
        }}

        /* ====================================================
           RECOMMENDATIONS
           ==================================================== */

        .recommendation-list {{
            display: grid;

            gap: 13px;
        }}

        .recommendation-card {{
            display: flex;

            gap: 15px;

            padding: 17px;

            border:
                1px solid var(--border);

            border-radius: 13px;

            background: #fcfcfd;
        }}

        .recommendation-number {{
            min-width: 38px;

            height: 38px;

            display: flex;

            align-items: center;

            justify-content: center;

            border-radius: 10px;

            background: #eef0ff;

            color: #5147d9;

            font-size: 12px;

            font-weight: 800;
        }}

        .recommendation-title {{
            font-weight: 700;

            margin-bottom: 3px;
        }}

        .recommendation-message {{
            color: var(--muted);

            font-size: 14px;
        }}

        /* ====================================================
           EMPTY
           ==================================================== */

        .empty-state {{
            display: flex;

            flex-direction: column;

            align-items: center;

            justify-content: center;

            padding: 45px 20px;

            color: var(--muted);

            text-align: center;
        }}

        .empty-icon {{
            display: flex;

            align-items: center;

            justify-content: center;

            width: 46px;

            height: 46px;

            margin-bottom: 10px;

            border-radius: 50%;

            background: #ecfdf3;

            color: var(--green);

            font-size: 22px;

            font-weight: 800;
        }}

        /* ====================================================
           FOOTER
           ==================================================== */

        footer {{
            text-align: center;

            padding-top: 20px;

            color: var(--muted);

            font-size: 13px;
        }}

        /* ====================================================
           RESPONSIVE
           ==================================================== */

        @media (max-width: 900px) {{

            .metrics {{
                grid-template-columns:
                    repeat(
                        2,
                        minmax(0, 1fr)
                    );
            }}

            .chart-grid {{
                grid-template-columns: 1fr;
            }}

            .issue-grid {{
                grid-template-columns: 1fr;
            }}

        }}

        @media (max-width: 600px) {{

            .container {{
                padding:
                    28px 14px 50px;
            }}

            header {{
                align-items: flex-start;

                gap: 15px;
            }}

            .brand-name {{
                font-size: 29px;
            }}

            .brand-icon {{
                font-size: 31px;
            }}

            .badge {{
                display: none;
            }}

            .metrics {{
                grid-template-columns: 1fr;
            }}

            .section,
            .health-card,
            .chart-card {{
                padding: 18px;
            }}

            .progress {{
                width: 90px;
            }}

        }}

    </style>

</head>


<body>


<div class="container">


    <!-- ====================================================
         HEADER
         ==================================================== -->

    <header>

        <div>

            <div class="brand">

                <span class="brand-icon">
                    🧬
                </span>

                <h1 class="brand-name">
                    DatasetDNA
                </h1>

            </div>

            <div class="subtitle">
                Automated Dataset Health Report
            </div>

        </div>

        <div class="badge">
            DATA QUALITY ANALYSIS
        </div>

    </header>


    <!-- ====================================================
         HEALTH SCORE
         ==================================================== -->

    <section class="card health-card">

        <div class="health-header">

            <h2>
                Dataset Health
            </h2>

            <div
                class="grade {_escape(score_class)}"
            >
                {_escape(grade)}
            </div>

        </div>

        {health_gauge}

    </section>


    <!-- ====================================================
         METRICS
         ==================================================== -->

    <div class="metrics">


        <div class="card metric">

            <div class="metric-label">
                ROWS
            </div>

            <div class="metric-value">
                {_escape(rows)}
            </div>

            <div class="metric-detail">
                Records analyzed
            </div>

        </div>


        <div class="card metric">

            <div class="metric-label">
                COLUMNS
            </div>

            <div class="metric-value">
                {_escape(columns)}
            </div>

            <div class="metric-detail">
                Features detected
            </div>

        </div>


        <div class="card metric">

            <div class="metric-label">
                MISSING CELLS
            </div>

            <div class="metric-value">
                {_escape(missing_cells)}
            </div>

            <div class="metric-detail">
                Across the dataset
            </div>

        </div>


        <div class="card metric">

            <div class="metric-label">
                OUTLIERS
            </div>

            <div class="metric-value">
                {_escape(outlier_count)}
            </div>

            <div class="metric-detail">
                Statistical anomalies
            </div>

        </div>


    </div>


    <!-- ====================================================
         DATA QUALITY CHARTS
         ==================================================== -->

    <div class="chart-grid">


        <section class="card chart-card">

            <div class="section-header">

                <h2 class="section-title">
                    📊 Missing Values
                </h2>

                <p class="section-description">
                    Columns containing missing data.
                </p>

            </div>

            {missing_chart}

        </section>


        <section class="card chart-card">

            <div class="section-header">

                <h2 class="section-title">
                    📌 Outlier Analysis
                </h2>

                <p class="section-description">
                    Percentage of statistical outliers.
                </p>

            </div>

            {outlier_chart}

        </section>


    </div>


    <!-- ====================================================
         NUMERICAL + TARGET
         ==================================================== -->

    <div class="chart-grid">


        <section class="card chart-card">

            <div class="section-header">

                <h2 class="section-title">
                    📈 Numerical Analysis
                </h2>

                <p class="section-description">
                    Mean versus median for numerical features.
                </p>

            </div>

            {numerical_chart}

        </section>


        <section class="card chart-card">

            <div class="section-header">

                <h2 class="section-title">
                    🎯 Target Analysis
                </h2>

                <p class="section-description">
                    Target class distribution.
                </p>

            </div>

            {target_chart}

        </section>


    </div>


    <!-- ====================================================
         CORRELATION
         ==================================================== -->

    <section class="card section">

        <div class="section-header">

            <h2 class="section-title">
                🔗 Correlation Analysis
            </h2>

            <p class="section-description">
                {correlation_count}
                numerical feature relationships detected.
            </p>

        </div>

        {correlation_chart}

    </section>


    <!-- ====================================================
         SCHEMA
         ==================================================== -->

    <section class="card section">

        <div class="section-header">

            <h2 class="section-title">
                🧱 Schema
            </h2>

            <p class="section-description">
                Detected data types for every column.
            </p>

        </div>

        <div class="table-wrapper">

            <table>

                <thead>

                    <tr>
                        <th>Column</th>
                        <th>Data Type</th>
                    </tr>

                </thead>

                <tbody>

                    {schema_rows}

                </tbody>

            </table>

        </div>

    </section>


    <!-- ====================================================
         MISSING DETAILS
         ==================================================== -->

    <section class="card section">

        <div class="section-header">

            <h2 class="section-title">
                Missing Value Details
            </h2>

            <p class="section-description">
                Missing-value distribution by column.
            </p>

        </div>

        <div class="table-wrapper">

            <table>

                <thead>

                    <tr>
                        <th>Column</th>
                        <th>Missing</th>
                        <th>Percentage</th>
                    </tr>

                </thead>

                <tbody>

                    {missing_rows}

                </tbody>

            </table>

        </div>

    </section>


    <!-- ====================================================
         CARDINALITY
         ==================================================== -->

    <section class="card section">

        <div class="section-header">

            <h2 class="section-title">
                🏷️ Cardinality
            </h2>

            <p class="section-description">
                Uniqueness and semantic type detection.
            </p>

        </div>

        <div class="table-wrapper">

            <table>

                <thead>

                    <tr>

                        <th>
                            Column
                        </th>

                        <th>
                            Unique Values
                        </th>

                        <th>
                            Unique %
                        </th>

                        <th>
                            Detected Type
                        </th>

                    </tr>

                </thead>

                <tbody>

                    {cardinality_rows}

                </tbody>

            </table>

        </div>

    </section>


    <!-- ====================================================
         CATEGORY CONSISTENCY
         ==================================================== -->

    <section class="card section">

        <div class="section-header">

            <h2 class="section-title">
                🏷️ Category Consistency
            </h2>

            <p class="section-description">
                Detects multiple representations of the same
                categorical value.
            </p>

        </div>

        {category_consistency_content}

    </section>


    <!-- ====================================================
         QUALITY ISSUES
         ==================================================== -->

    <section class="card section">

        <div class="section-header">

            <h2 class="section-title">
                ⚠️ Quality Issues
            </h2>

            <p class="section-description">
                {len(issues)}
                data-quality issue(s) detected.
            </p>

        </div>

        <div class="issue-grid">

            {issue_cards}

        </div>

    </section>


    <!-- ====================================================
         RECOMMENDATIONS
         ==================================================== -->

    <section class="card section">

        <div class="section-header">

            <h2 class="section-title">
                💡 Recommendations
            </h2>

            <p class="section-description">
                Actionable suggestions generated from the analysis.
            </p>

        </div>

        <div class="recommendation-list">

            {recommendation_cards}

        </div>

    </section>


    <!-- ====================================================
         FOOTER
         ==================================================== -->

    <footer>

        DatasetDNA · Automated Dataset Intelligence

    </footer>


</div>


</body>

</html>
"""

    # ========================================================
    # WRITE REPORT
    # ========================================================

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(document)

    return output_path