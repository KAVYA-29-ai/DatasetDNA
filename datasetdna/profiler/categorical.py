import pandas as pd


def check_categorical(df: pd.DataFrame) -> dict:
    """
    Analyze categorical/string columns.

    This function returns raw category frequencies.
    It does not decide whether a category is rare.
    """

    result = {}

    categorical_df = df.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    )

    for column in categorical_df.columns:

        series = categorical_df[column].dropna()

        if series.empty:
            result[column] = {
                "unique_count": 0,
                "total_values": 0,
                "categories": {},
            }
            continue

        value_counts = series.value_counts()

        total_values = len(series)

        categories = {}

        for category, count in value_counts.items():

            percentage = (
                count / total_values
            ) * 100

            categories[str(category)] = {
                "count": int(count),
                "percentage": round(
                    float(percentage),
                    2
                ),
            }

        result[column] = {
            "unique_count": int(
                series.nunique()
            ),
            "total_values": int(
                total_values
            ),
            "categories": categories,
        }

    return result