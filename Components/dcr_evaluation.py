import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors


def compute_dcr(container_input_real, container_input_synth, container_output):
    """
    Distance to Closest Record (DCR)

    Computes the distance from each synthetic record to its closest real record.

    This component reports objective distance statistics only:
      - minimum DCR
      - mean DCR
      - median DCR
      - maximum DCR
      - standard deviation of DCR

    No threshold or risk level is applied because DCR scale depends on
    the dataset dimensionality, feature types, encoding, and scaling.
    """

    df_real = pd.read_parquet(container_input_real)
    df_synth = pd.read_parquet(container_input_synth)

    common_cols = df_real.columns.intersection(df_synth.columns)

    if len(common_cols) == 0:
        raise ValueError("No common columns found between real and synthetic datasets.")

    df_real = df_real[common_cols].copy()
    df_synth = df_synth[common_cols].copy()

    df_real = df_real.dropna()
    df_synth = df_synth.dropna()

    if df_real.empty:
        raise ValueError("Real dataset is empty after removing missing values.")

    if df_synth.empty:
        raise ValueError("Synthetic dataset is empty after removing missing values.")

    combined = pd.concat([df_real, df_synth], ignore_index=True)

    categorical_columns = combined.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numerical_columns = combined.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

    print("========== DCR Evaluation ==========")
    print(f"Common columns used: {list(common_cols)}")
    print(f"Categorical columns: {categorical_columns}")
    print(f"Numerical columns: {numerical_columns}")
    print("Distance metric: euclidean")

    preprocessor = ColumnTransformer(transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns), ("numerical", StandardScaler(), numerical_columns)])

    preprocessor.fit(combined)

    real_encoded = preprocessor.transform(df_real)
    synth_encoded = preprocessor.transform(df_synth)

    nearest_model = NearestNeighbors(n_neighbors=1, metric="euclidean")

    nearest_model.fit(real_encoded)

    distances, _ = nearest_model.kneighbors(synth_encoded)

    distances = distances.flatten()

    dcr_min = float(np.min(distances))
    dcr_mean = float(np.mean(distances))
    dcr_median = float(np.median(distances))
    dcr_max = float(np.max(distances))
    dcr_std = float(np.std(distances))

    summary = pd.DataFrame([{
        "metric_name": "Distance to Closest Record (DCR)",
        "method": "Nearest-neighbour Euclidean distance from each synthetic record to the closest real record",
        "distance_metric": "euclidean",
        "common_columns_used": ",".join(common_cols),
        "categorical_columns": ",".join(categorical_columns),
        "numerical_columns": ",".join(numerical_columns),
        "real_records_used": len(df_real),
        "synthetic_records_used": len(df_synth),
        "dcr_min": round(dcr_min, 6),
        "dcr_mean": round(dcr_mean, 6),
        "dcr_median": round(dcr_median, 6),
        "dcr_max": round(dcr_max, 6),
        "dcr_std": round(dcr_std, 6),
        "interpretation": (
            "DCR measures how close synthetic records are to real records. "
            "Very small values may indicate that some synthetic records are "
            "near copies of real records. DCR should be interpreted together "
            "with other privacy and utility metrics."
        )
    }])

    summary.to_parquet(container_output, index=False)

    print("✅ DCR evaluation completed successfully.")
    print(summary)


if __name__ == "__main__":
    container_input_real = os.environ.get("container_input_real")
    container_input_synth = os.environ.get("container_input_synth")
    container_output = os.environ.get("container_output")

    if not container_input_real:
        raise ValueError("container_input_real is required.")

    if not container_input_synth:
        raise ValueError("container_input_synth is required.")

    if not container_output:
        raise ValueError("container_output is required.")

    compute_dcr(
        container_input_real=container_input_real,
        container_input_synth=container_input_synth,
        container_output=container_output
    )
