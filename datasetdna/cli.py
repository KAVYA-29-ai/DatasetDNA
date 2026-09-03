import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from datasetdna.profiler.overview import check_overview
from datasetdna.profiler.schema import check_schema
from datasetdna.profiler.missing import check_missing

app = typer.Typer()
console = Console()


@app.command()
def profile(
    file: str,
    target: str = typer.Option(None, "--target")
):
    """Profile a CSV dataset."""

    try:
        df = pd.read_csv(file)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    overview = check_overview(df)
    schema = check_schema(df)
    missing = check_missing(df)

    console.print("\n[bold cyan]🧬 DatasetDNA[/bold cyan]")
    console.print("[bold]Dataset Health Report[/bold]\n")

    table = Table(title="Dataset Overview")

    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Rows", str(overview["rows"]))
    table.add_row("Columns", str(overview["columns"]))
    table.add_row("Memory", f'{overview["memory_usage_mb"]} MB')
    table.add_row(
        "Missing Cells",
        str(missing["total_missing_cells"])
    )

    console.print(table)

    console.print("\n[bold]Schema[/bold]")

    schema_table = Table()

    schema_table.add_column("Column")
    schema_table.add_column("Data Type")

    for column, dtype in schema.items():
        schema_table.add_row(column, dtype)

    console.print(schema_table)

    if target:
        if target not in df.columns:
            console.print(
                f"[red]Target column '{target}' not found.[/red]"
            )
            raise typer.Exit(code=1)

        console.print(
            f"\n[bold yellow]Target:[/bold yellow] {target}"
        )


if __name__ == "__main__":
    app()
