from __future__ import annotations

import os

import typer

from datasetdna.engine.profiler import profile_dataframe

from datasetdna.reporting.console import (
    console,
    render_report,
)

from datasetdna.reporting.html import (
    render_html_report,
)

from datasetdna.utils.helpers import (
    LARGE_FILE_SIZE_BYTES,
    LARGE_FILE_SAMPLE_SIZE,
    load_dataset,
)


app = typer.Typer(
    help="DatasetDNA - Automated Dataset Health Profiler",
    add_completion=False,
)


# =============================================================
# LARGE FILE WARNING
# =============================================================

def warn_if_large_file(
    path: str,
) -> None:
    """
    Warn the user when DatasetDNA will analyze a sample
    instead of the complete dataset.
    """

    if not os.path.exists(path):
        return

    if os.path.getsize(path) > LARGE_FILE_SIZE_BYTES:
        typer.echo(
            f"Large file detected — analyzing a "
            f"{LARGE_FILE_SAMPLE_SIZE:,}-row sample."
        )


# =============================================================
# CLI COMMAND
# =============================================================

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

        # =====================================================
        # LARGE FILE WARNING
        # =====================================================

        warn_if_large_file(file)

        # =====================================================
        # LOAD DATASET
        # =====================================================

        df = load_dataset(file)

        # =====================================================
        # DATASETDNA ENGINE
        # =====================================================

        results = profile_dataframe(
            df,
            target=target,
        )

        # =====================================================
        # AUTO-INFERRED TARGET MESSAGE
        # =====================================================

        target_result = results.get(
            "target",
            {},
        )

        if target_result.get("inferred"):
            inferred_target = target_result.get(
                "column"
            )

            console.print(
                f"[yellow]ℹ Auto-inferred "
                f"'{inferred_target}' as the primary "
                f"target variable.[/yellow]\n"
            )

        # =====================================================
        # HEALTH SCORE
        # =====================================================

        health = results["health"]

        # =====================================================
        # RECOMMENDATIONS
        # =====================================================

        recommendations = results[
            "recommendations"
        ]

        # =====================================================
        # CONSOLE REPORT
        # =====================================================

        render_report(
            results,
            health,
            recommendations,
        )

        # =====================================================
        # HTML REPORT
        # =====================================================

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

        raise typer.Exit(
            code=1
        )

    except ValueError as error:

        typer.echo(
            f"Error: {error}",
            err=True,
        )

        raise typer.Exit(
            code=1
        )

    except Exception as error:

        typer.echo(
            f"Unexpected error: {error}",
            err=True,
        )

        raise typer.Exit(
            code=1
        )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    app()