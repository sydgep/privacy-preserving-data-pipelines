import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


# Fixed histogram resolution for numerical Jensen–Shannon distance.
# This is intentionally hardcoded and is not exposed as a component parameter.
JS_BINS = 30

NUMERICAL_INTERPRETATION = (
    "Values closer to 0 indicate greater numerical fidelity. "
    "KS measures the maximum difference between the real and synthetic "
    "empirical cumulative distributions. Wasserstein measures distribution "
    "displacement normalized by the real-data range. JS distance compares "
    "the real and synthetic distributions using shared histogram intervals."
)

CATEGORICAL_INTERPRETATION = (
    "JS distance compares the real and synthetic category-frequency "
    "distributions over the union of observed categories. "
    "A value of 0 indicates identical category proportions; larger values "
    "indicate greater distributional difference. KS and Wasserstein are not "
    "applicable to categorical variables."
)

INCOMPATIBLE_INTERPRETATION = (
    "The column was not evaluated because its real and synthetic "
    "representations have incompatible data types."
)

EMPTY_INTERPRETATION = (
    "The column was not evaluated because no valid non-missing values were "
    "available in one or both datasets."
)


def validate_environment_variables(
    container_input_real,
    container_input_synth,
    container_output,
):
    """
    Validate required environment variables while retaining os.environ.get().
    """
    missing_variables = []

    if not container_input_real:
        missing_variables.append("container_input_real")

    if not container_input_synth:
        missing_variables.append("container_input_synth")

    if not container_output:
        missing_variables.append("container_output")

    if missing_variables:
        raise ValueError(
            "Missing required environment variable(s): "
            + ", ".join(missing_variables)
        )


def load_dataset(file_path):
    """
    Load a supported tabular dataset.

    Supported formats:
    - Parquet
    - CSV
    - CSV.GZ
    """
    lower_path = file_path.lower()

    if lower_path.endswith(".parquet"):
        return pd.read_parquet(file_path)

    if lower_path.endswith(".csv") or lower_path.endswith(".csv.gz"):
        return pd.read_csv(file_path)

    raise ValueError(
        f"Unsupported input format for '{file_path}'. "
        "Supported formats are .parquet, .csv and .csv.gz."
    )


def save_results(df_result, output_path):
    """
    Save results according to the requested output extension.
    """
    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    lower_path = output_path.lower()

    if lower_path.endswith(".parquet"):
        df_result.to_parquet(output_path, index=False)
        return

    if lower_path.endswith(".csv"):
        df_result.to_csv(output_path, index=False)
        return

    raise ValueError(
        f"Unsupported output format for '{output_path}'. "
        "Use either .parquet or .csv."
    )


def convert_to_numeric(series):
    """
    Attempt to convert a series to numeric values.

    Returns:
        converted_series: Numeric pandas Series.
        conversion_successful: True when every non-missing original value
                               can be converted to a numeric value.
    """
    original_non_missing = series.dropna()

    converted_series = pd.to_numeric(series, errors="coerce")
    converted_non_missing = converted_series.dropna()

    conversion_successful = (
        len(original_non_missing) == len(converted_non_missing)
    )

    return converted_series, conversion_successful


def determine_data_type(real_series, synth_series):
    """
    Determine whether a matching column can be evaluated as numerical
    or categorical.

    Rules:
    - Numeric in both datasets: numerical.
    - Numeric-looking strings in both datasets: numerical.
    - Non-numeric in both datasets: categorical.
    - Numeric in one dataset and non-numeric in the other: incompatible.
    """
    real_numeric, real_numeric_compatible = convert_to_numeric(real_series)
    synth_numeric, synth_numeric_compatible = convert_to_numeric(synth_series)

    if real_numeric_compatible and synth_numeric_compatible:
        return "numerical", real_numeric, synth_numeric

    if not real_numeric_compatible and not synth_numeric_compatible:
        return "categorical", real_series, synth_series

    return "incompatible", real_series, synth_series


def clean_numeric_values(series):
    """
    Remove missing and infinite numerical values.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def compute_numerical_js_distance(real_values, synth_values):
    """
    Compute Jensen–Shannon distance for numerical distributions.

    Real and synthetic histograms use the same bin edges derived from
    their combined value range.
    """
    combined_values = np.concatenate([real_values, synth_values])

    # Both distributions contain the same single constant value.
    if np.all(combined_values == combined_values[0]):
        return 0.0

    bin_edges = np.histogram_bin_edges(
        combined_values,
        bins=JS_BINS,
    )

    real_hist, _ = np.histogram(
        real_values,
        bins=bin_edges,
    )

    synth_hist, _ = np.histogram(
        synth_values,
        bins=bin_edges,
    )

    real_probabilities = real_hist.astype(float)
    synth_probabilities = synth_hist.astype(float)

    real_probabilities /= real_probabilities.sum()
    synth_probabilities /= synth_probabilities.sum()

    return float(
        jensenshannon(
            real_probabilities,
            synth_probabilities,
            base=2,
        )
    )


def compute_categorical_js_distance(real_series, synth_series):
    """
    Compute Jensen–Shannon distance between categorical frequency
    distributions.

    Categories are aligned using the union of categories observed in
    the real and synthetic datasets.
    """
    real_values = real_series.dropna().astype(str)
    synth_values = synth_series.dropna().astype(str)

    categories = sorted(
        set(real_values.unique()).union(synth_values.unique())
    )

    real_probabilities = (
        real_values.value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
        .to_numpy(dtype=float)
    )

    synth_probabilities = (
        synth_values.value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
        .to_numpy(dtype=float)
    )

    return float(
        jensenshannon(
            real_probabilities,
            synth_probabilities,
            base=2,
        )
    )


def compute_normalized_wasserstein(real_values, synth_values):
    """
    Compute Wasserstein distance normalized by the real-data range.

    normalized Wasserstein =
        raw Wasserstein / (real maximum - real minimum)

    Constant real columns:
    - Return 0 when the distributions are identical.
    - Return None when the distributions differ, because normalization
      by a zero real-data range is undefined.
    """
    raw_wasserstein = float(
        wasserstein_distance(real_values, synth_values)
    )

    real_range = float(
        np.max(real_values) - np.min(real_values)
    )

    if real_range == 0:
        if raw_wasserstein == 0:
            return 0.0

        return None

    return raw_wasserstein / real_range


def evaluate_numerical_column(column, real_series, synth_series):
    """
    Evaluate one numerical column.
    """
    real_values = clean_numeric_values(real_series)
    synth_values = clean_numeric_values(synth_series)

    if len(real_values) == 0 or len(synth_values) == 0:
        return {
            "column": column,
            "data_type": "numerical",
            "ks_stat": None,
            "wasserstein": None,
            "js_distance": None,
            "interpretation": EMPTY_INTERPRETATION,
        }

    ks_statistic = float(
        ks_2samp(
            real_values,
            synth_values,
        ).statistic
    )

    normalized_wasserstein = compute_normalized_wasserstein(
        real_values,
        synth_values,
    )

    js_distance = compute_numerical_js_distance(
        real_values,
        synth_values,
    )

    return {
        "column": column,
        "data_type": "numerical",
        "ks_stat": round(ks_statistic, 4),
        "wasserstein": (
            round(normalized_wasserstein, 4)
            if normalized_wasserstein is not None
            else None
        ),
        "js_distance": round(js_distance, 4),
        "interpretation": NUMERICAL_INTERPRETATION,
    }


def evaluate_categorical_column(column, real_series, synth_series):
    """
    Evaluate one categorical column.
    """
    real_values = real_series.dropna()
    synth_values = synth_series.dropna()

    if real_values.empty or synth_values.empty:
        return {
            "column": column,
            "data_type": "categorical",
            "ks_stat": None,
            "wasserstein": None,
            "js_distance": None,
            "interpretation": EMPTY_INTERPRETATION,
        }

    js_distance = compute_categorical_js_distance(
        real_values,
        synth_values,
    )

    return {
        "column": column,
        "data_type": "categorical",
        "ks_stat": None,
        "wasserstein": None,
        "js_distance": round(js_distance, 4),
        "interpretation": CATEGORICAL_INTERPRETATION,
    }


def statistical_evaluation(
    container_input_real,
    container_input_synth,
    container_output,
):
    """
    Evaluate univariate statistical fidelity between real and synthetic
    tabular datasets.
    """
    df_real = load_dataset(container_input_real)
    df_synth = load_dataset(container_input_synth)

    common_columns = [
        column
        for column in df_real.columns
        if column in df_synth.columns
    ]

    if not common_columns:
        raise ValueError(
            "No matching columns were found in the real and synthetic datasets."
        )

    results = []

    for column in common_columns:
        real_series = df_real[column]
        synth_series = df_synth[column]

        data_type, prepared_real, prepared_synth = determine_data_type(
            real_series,
            synth_series,
        )

        if data_type == "numerical":
            result = evaluate_numerical_column(
                column,
                prepared_real,
                prepared_synth,
            )

        elif data_type == "categorical":
            result = evaluate_categorical_column(
                column,
                prepared_real,
                prepared_synth,
            )

        else:
            result = {
                "column": column,
                "data_type": "incompatible",
                "ks_stat": None,
                "wasserstein": None,
                "js_distance": None,
                "interpretation": INCOMPATIBLE_INTERPRETATION,
            }

        results.append(result)

    df_result = pd.DataFrame(
        results,
        columns=[
            "column",
            "data_type",
            "ks_stat",
            "wasserstein",
            "js_distance",
            "interpretation",
        ],
    )

    save_results(
        df_result,
        container_output,
    )

    print("Statistical fidelity evaluation completed.")
    print(df_result.to_string(index=False))


if __name__ == "__main__":
    container_input_real = os.environ.get("container_input_real")
    container_input_synth = os.environ.get("container_input_synth")
    container_output = os.environ.get("container_output")

    validate_environment_variables(container_input_real, container_input_synth, container_output,)

    statistical_evaluation(
        container_input_real=container_input_real,
        container_input_synth=container_input_synth,
        container_output=container_output,
    )
