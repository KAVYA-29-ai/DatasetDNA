import pandas as pd
import typer

from datasetdna.profiler.overview import check_overview
from datasetdna.profiler.schema import check_schema
from datasetdna.profiler.missing import check_missing
from datasetdna.profiler.duplicates import check_duplicates
from datasetdna.profiler.cardinality import check_cardinality
from datasetdna.profiler.numerical import check_numerical
from datasetdna.profiler.categorical import check_categorical
from datasetdna.profiler.outliers import check_outliers
from datasetdna.profiler.target import check_target
from datasetdna.profiler.correlations import check_correlations

from datasetdna.scoring.health_score import calculate_health_score

from datasetdna.reporting.console import render_report


app = typer.Typer(
    help="DatasetDNA - Automated Dataset Health Profiler",
    add_completion=False,
)


@app.command()
def profile(
    file: str = typer.Argument(
        ...,
        help="Path to the CSV dataset.",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Target column for target analysis.",
    ),
):
    """
    Profile a CSV dataset and generate a DatasetDNA health report.
    """

    # =========================================================
    # LOAD DATASET
    # =========================================================

    try:
        df = pd.read_csv(file)

    except FileNotFoundError:
        print(f"Error: File not found: {file}")
        raise typer.Exit(code=1)

    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        raise typer.Exit(code=1)

    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        raise typer.Exit(code=1)

    # =========================================================
    # RUN PROFILERS
    # =========================================================

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

    # =========================================================
    # COLLECT RAW RESULTS
    # =========================================================

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

    # =========================================================
    # CALCULATE HEALTH SCORE
    # =========================================================

    health = calculate_health_score(
        results
    )

    # =========================================================
    # RENDER REPORT
    # =========================================================

    render_report(
        results,
        health,
    )


if __name__ == "__main__":
    app()