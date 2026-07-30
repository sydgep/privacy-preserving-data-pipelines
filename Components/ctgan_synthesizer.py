import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


def ctgan_synthesizer(container_input, container_output):
    """
    Train a CTGAN model on the input table and generate synthetic data.
    Saves both Parquet and CSV versions of the synthetic dataset.
    """
    # ---- Load input --------------------------------------------------------
    df = pd.read_parquet(container_input)

    # ---- Build metadata ----------------------------------------------------
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # ---- Instantiate & fit CTGAN -------------------------------------------
    ctgan = CTGANSynthesizer(metadata)
    ctgan.fit(df)

    # ---- Sample synthetic data ---------------------------------------------
    df_synth = ctgan.sample(len(df))

    # Save parquet
    df_synth.to_parquet(container_output, index=False)

    # Save csv (same name but with .csv extension)
    csv_output = os.path.splitext(container_output)[0] + ".csv"
    df_synth.to_csv(csv_output, index=False)

    print("✅ Synthetic data written →", container_output)
    print("✅ Synthetic data written →", csv_output)


if __name__ == "__main__":
    container_input  = os.environ.get("container_input")
    container_output = os.environ.get("container_output")

    ctgan_synthesizer(container_input, container_output)

