import json
import os
import pandas as pd


def direct_identifier_removal(
    container_input: str,
    container_output: str,
    identifier_columns: list
):
    """
    Removes user-specified direct identifier columns from a dataset.

    Parameters
    ----------
    container_input : str
        Path to the input Parquet dataset.

    container_output : str
        Path where the processed dataset will be written.

    identifier_columns : list
        List of direct identifier columns to remove.

    Notes
    -----
    - No automatic identifier detection is performed.
    - Columns that do not exist are ignored with a warning.
    - Remaining columns are preserved unchanged.
    """

    df = pd.read_parquet(container_input)

    print("\n========== Direct Identifier Removal ==========")

    print(f"Input dataset shape : {df.shape}")
    print(f"Input columns       : {df.columns.tolist()}")

    print(f"\nRequested identifier columns:")
    print(identifier_columns)

    existing_columns = [
        column
        for column in identifier_columns
        if column in df.columns
    ]

    missing_columns = [
        column
        for column in identifier_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nWarning:")
        print(
            "The following columns were not found "
            "and will be ignored:"
        )
        print(missing_columns)

    if existing_columns:
        df = df.drop(columns=existing_columns)

        print("\nRemoved columns:")
        print(existing_columns)
    else:
        print("\nNo columns were removed.")

    print(f"\nOutput dataset shape : {df.shape}")
    print(f"Output columns       : {df.columns.tolist()}")

    df.to_parquet(container_output, index=False)
    
    # Save to csv with same name but .csv extension
    csv_output = os.path.splitext(container_output)[0] + ".csv"
    df.to_csv(csv_output, index=False)

    print("\nDirect Identifier Removal completed successfully.")


if __name__ == "__main__":

    container_input = os.environ.get("container_input")
    container_output = os.environ.get("container_output")

    identifier_columns_raw = os.environ.get("identifier_columns", "[]")

    if not container_input:
        raise ValueError("container_input environment variable is required.")

    if not container_output:
        raise ValueError("container_output environment variable is required.")

    identifier_columns = json.loads(identifier_columns_raw)

    if not isinstance(identifier_columns, list):
        raise ValueError("identifier_columns must be a JSON list.")

    direct_identifier_removal(
        container_input=container_input,
        container_output=container_output,
        identifier_columns=identifier_columns
    )
    
    
