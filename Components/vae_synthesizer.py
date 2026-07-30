import os

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer


TABLE_NAME = "input_table"
TVAE_EPOCHS = 500


def load_dataset(file_path):
    """
    Load a supported tabular dataset.
    """
    lower_path = file_path.lower()

    if lower_path.endswith(".parquet"):
        return pd.read_parquet(file_path)

    if lower_path.endswith(".csv") or lower_path.endswith(".csv.gz"):
        return pd.read_csv(file_path)

    raise ValueError(
        f"Unsupported input format: {file_path}. "
        "Supported formats are .parquet, .csv and .csv.gz."
    )


def validate_inputs(container_input, container_output):
    """
    Validate the required container inputs.
    """
    if not container_input:
        raise ValueError(
            "The environment variable 'container_input' is required."
        )

    if not container_output:
        raise ValueError(
            "The environment variable 'container_output' is required."
        )

    if not os.path.isfile(container_input):
        raise FileNotFoundError(
            f"Input dataset not found: {container_input}"
        )


def detect_and_update_metadata(df):
    """
    Automatically detect SDV metadata and ensure suitable text-like
    columns are treated as categorical.

    No dataset-specific column names are used.
    """
    metadata = Metadata.detect_from_dataframe(
        data=df,
        table_name=TABLE_NAME,
        infer_sdtypes=True,
        infer_keys=None,
    )

    for column in df.columns:
        series = df[column]

        # Object, string and categorical pandas columns normally
        # represent discrete values in tabular datasets.
        is_text_like = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

        if is_text_like:
            metadata.update_column(
                table_name=TABLE_NAME,
                column_name=column,
                sdtype="categorical",
            )

    metadata.validate()

    return metadata


def print_detected_metadata(metadata):
    """
    Print the detected SDV type for each column.
    """
    metadata_dict = metadata.to_dict()
    columns = metadata_dict["tables"][TABLE_NAME]["columns"]

    print("\nDetected column types:")

    for column, definition in columns.items():
        print(
            f"  {column}: {definition.get('sdtype', 'unknown')}"
        )


def print_collapsed_columns(df_real, df_synth):
    """
    Identify categorical columns that contain multiple values in the
    real data but only one value in the synthetic data.

    This is diagnostic only and does not alter the generated data.
    """
    print("\nCategorical support check:")

    collapsed_columns = []

    for column in df_real.columns:
        series = df_real[column]

        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

        if not is_categorical:
            continue

        real_unique = set(
            series.dropna().astype(str).unique()
        )

        synthetic_unique = set(
            df_synth[column].dropna().astype(str).unique()
        )

        missing_values = sorted(
            real_unique - synthetic_unique
        )

        if len(real_unique) > 1 and len(synthetic_unique) == 1:
            collapsed_columns.append(column)

            print(
                f"WARNING: '{column}' collapsed from "
                f"{len(real_unique)} real values to one synthetic value: "
                f"{sorted(synthetic_unique)}"
            )

        elif missing_values:
            print(
                f"WARNING: '{column}' is missing the following "
                f"real values in the synthetic data: {missing_values}"
            )

        else:
            print(
                f"'{column}': all real categories are represented."
            )

    if not collapsed_columns:
        print("No categorical column collapse was detected.")


def vae_synthesizer(container_input, container_output):
    """
    Train a generic TVAE model and generate synthetic tabular data.

    The function retains only the original two input parameters:
    container_input and container_output.
    """
    validate_inputs(
        container_input=container_input,
        container_output=container_output,
    )

    # ---------------------------------------------------------
    # 1. Load the real dataset
    # ---------------------------------------------------------
    df = load_dataset(container_input)

    if df.empty:
        raise ValueError("The input dataset is empty.")

    print(f"Input dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Automatically detect metadata
    # ---------------------------------------------------------
    metadata = detect_and_update_metadata(df)

    print_detected_metadata(metadata)

    # ---------------------------------------------------------
    # 3. Prepare output paths
    # ---------------------------------------------------------
    output_directory = os.path.dirname(container_output)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    output_base = os.path.splitext(container_output)[0]

    csv_output = output_base + ".csv"
    metadata_output = output_base + "_metadata.json"
    loss_output = output_base + "_loss_values.csv"

    # Saving metadata removes the reproducibility warning and allows
    # future inspection of the detected column types.
    metadata.save_to_json(
        filepath=metadata_output,
        mode="overwrite",
    )

    # ---------------------------------------------------------
    # 4. Train TVAE
    # ---------------------------------------------------------
    synthesizer = TVAESynthesizer(
        metadata=metadata,
        epochs=TVAE_EPOCHS,
        verbose=True,
        enforce_min_max_values=True,
        enforce_rounding=False,
    )

    synthesizer.fit(data=df)

    # ---------------------------------------------------------
    # 5. Save training loss
    # ---------------------------------------------------------
    loss_values = synthesizer.get_loss_values()

    loss_values.to_csv(
        loss_output,
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Generate synthetic data
    # ---------------------------------------------------------
    df_synth = synthesizer.sample(
        num_rows=len(df)
    )

    print(f"\nSynthetic dataset shape: {df_synth.shape}")

    # Automatically report categorical collapse for every
    # categorical column without requiring a target parameter.
    print_collapsed_columns(
        df_real=df,
        df_synth=df_synth,
    )

    # ---------------------------------------------------------
    # 7. Save synthetic datasets
    # ---------------------------------------------------------
    df_synth.to_parquet(container_output, index=False,)

    df_synth.to_csv(csv_output, index=False,)

    print("\nSynthetic data written to:")
    print(container_output)
    print(csv_output)

    print("\nMetadata written to:")
    print(metadata_output)

    print("\nTVAE loss values written to:")
    print(loss_output)


if __name__ == "__main__":
    container_input = os.environ.get("container_input")
    container_output = os.environ.get("container_output")

    vae_synthesizer(
        container_input=container_input,
        container_output=container_output,
    )
