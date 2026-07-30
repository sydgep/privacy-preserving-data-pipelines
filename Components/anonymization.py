import os
import pandas as pd
import numpy as np


def numeric_interval_generalization(series, interval):
    """
    Generalize numeric values into intervals.

    Example:
        interval=5

        32 -> 30-34
        35 -> 35-39
    """

    def generalize(value):

        if pd.isna(value):
            return value

        lower = int(np.floor(float(value) / interval) * interval)
        upper = lower + interval - 1

        return f"{lower}-{upper}"

    return series.apply(generalize)


def date_generalization(series, precision):
    """
    Generalize dates.

    Supported precisions:
        year
        month
        day
    """

    series = pd.to_datetime(series)

    if precision == "year":
        return series.dt.strftime("%Y")

    elif precision == "month":
        return series.dt.strftime("%Y-%m")

    elif precision == "day":
        return series.dt.strftime("%Y-%m-%d")

    else:
        raise ValueError(
            f"Unsupported precision: {precision}"
        )


def anonymization(
    container_input,
    container_output,
    column,
    strategy,
    interval=None,
    precision=None
):

    df = pd.read_parquet(container_input)

    print("\n========== ANONYMIZATION ==========")

    print(f"Input dataset shape : {df.shape}")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found."
        )

    print(f"\nColumn   : {column}")
    print(f"Strategy : {strategy}")

    if strategy == "numeric_interval":

        if interval is None:
            raise ValueError(
                "interval is required "
                "for numeric_interval."
            )

        df[column] = numeric_interval_generalization(
            df[column],
            int(interval)
        )

    elif strategy == "date":

        if precision is None:
            raise ValueError(
                "precision is required "
                "for date strategy."
            )

        df[column] = date_generalization(
            df[column],
            precision
        )

    else:

        raise ValueError(f"Unsupported strategy '{strategy}'.")

    print(f"\nOutput dataset shape : {df.shape}")

    df.to_parquet(container_output,  index=False)
    
    # Save to csv with same name but .csv extension
    csv_output = os.path.splitext(container_output)[0] + ".csv"
    df.to_csv(csv_output, index=False)

    print("\nAnonymization completed successfully.")


if __name__ == "__main__":

    container_input = os.environ.get("container_input")
    container_output = os.environ.get("container_output")

    column = os.environ.get("column")
    strategy = os.environ.get("strategy")

    interval = os.environ.get("interval")
    precision = os.environ.get("precision")

    if not container_input:
        raise ValueError("container_input is required.")

    if not container_output:
        raise ValueError("container_output is required.")

    if not column:
        raise ValueError("column is required.")

    if not strategy:
        raise ValueError("strategy is required.")

    anonymization(
        container_input=container_input,
        container_output=container_output,
        column=column,
        strategy=strategy,
        interval=interval,
        precision=precision
    )
