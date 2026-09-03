from __future__ import annotations

import typer

from datasetdna.profiler.overview import check_overview
from datasetdna.profiler.schema import check_schema
from datasetdna.profiler.missing import check_missing
from datasetdna.profiler.duplicates import check_duplicates
from datasetdna.profiler.cardinality import check_cardinality
from datasetdna.profiler.numerical import check_numerical
from datasetdna.profiler.categorical import check_categorical
from datasetdna.profiler.outliers import check_outliers
from datasetdna.profiler.correlations import check_correlations
from datasetdna.profiler.target import check_target

from datasetdna.scoring.health_score import (
    calculate_health_score,
)

from datasetdna.recommendations.recommendations import (
    generate_recommendations,
)

from datasetdna.reporting.console import (
    render_report,
)

from datasetdna.reporting.html import (
    render_html_report,
)

from datasetdna.utils.helpers import (
    load_dataset,
)


app = typer.Typer(
    help="DatasetDNA - Automated Dataset Health Profiler",
    add_completion=False,
)


@app.command()
def profile(
    file: str = typer.Argument(
        ...,
        help="Path to the CSV file.",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Optional target column.",
    ),
    html: bool = typer.Option(
        False,
        "--html",
        help="Generate an HTML report.",
    ),
    output: str = typer.Option(
        "datasetdna_report.html",
        "--output",
        "-o",
        help="HTML output file path.",
    ),
):
    """
    Profile a CSV dataset and generate a health report.
    """

    try:
        # ====================================================
        # LOAD DATASET
        # ====================================================

        df = load_dataset(file)

        # ====================================================
        # RUN PROFILERS
        # ====================================================

        overview = check_overview(df)

        schema = check_schema(df)

        missing = check_missing(df)

        duplicates = check_duplicates(df)

        cardinality = check_cardinality(df)

        numerical = check_numerical(df)

        categorical = check_categorical(df)

        outliers = check_outliers(df)

        correlations = check_correlations(df)

        target_result = check_target(
            df,
            target,
        )

        # ====================================================
        # COLLECT RESULTS
        # ====================================================

        results = {
            "overview": overview,
            "schema": schema,
            "missing": missing,
            "duplicates": duplicates,
            "cardinality": cardinality,
            "numerical": numerical,
            "categorical": categorical,
            "outliers": outliers,
            "correlations": correlations,
            "target": target_result,
        }

        # ====================================================
        # HEALTH SCORE
        # ====================================================

        health = calculate_health_score(
            results
        )

        results["health"] = health

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        recommendations = generate_recommendations(
            results
        )

        results["recommendations"] = recommendations

        # ====================================================
        # CONSOLE REPORT
        # ====================================================

        render_report(
            results,
            health,
            recommendations,
        )

        # ====================================================
        # HTML REPORT
        # ====================================================

        if html:
            output_path = render_html_report(
                results,
                output_path=output,
            )

            typer.echo(
                f"\nHTML report generated: {output_path}"
            )

    except FileNotFoundError as error:
        typer.echo(
            f"Error: {error}",
            err=True,
        )
        raise typer.Exit(code=1)

    except ValueError as error:
        typer.echo(
            f"Error: {error}",
            err=True,
        )
        raise typer.Exit(code=1)

    except Exception as error:
        typer.echo(
            f"Unexpected error: {error}",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()