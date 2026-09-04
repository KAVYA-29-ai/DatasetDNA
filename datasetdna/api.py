from __future__ import annotations

import pandas as pd

from datasetdna.engine.profiler import profile_dataframe
from datasetdna.reporting.html import render_html_report


def profile(
    df: pd.DataFrame,
    target: str | None = None,
    html: bool = False,
    output: str = "datasetdna_report.html",
) -> dict:
    """
    Profile an in-memory pandas DataFrame.

    Parameters
    ----------
    df:
        DataFrame to analyze.

    target:
        Optional target column.

    html:
        Generate an interactive HTML report when True.

    output:
        HTML output path.
    """

    results = profile_dataframe(
        df,
        target=target,
    )

    if html:
        render_html_report(
            results,
            output_path=output,
        )

    return results