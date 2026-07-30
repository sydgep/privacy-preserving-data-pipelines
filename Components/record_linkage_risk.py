import os
import json
import pandas as pd
import numpy as np


def normalize_quasi_identifier_values(df, quasi_identifier_columns):
    """
    Normalize quasi-identifier values for exact matching.

    This does not anonymize or transform the dataset permanently.
    It only creates a temporary matching representation.

    - Converts values to string
    - Strips leading/trailing whitespace
    - Converts to lowercase
    - Replaces missing values with a fixed token
    """

    df_match = df[quasi_identifier_columns].copy()

    for col in quasi_identifier_columns:
        df_match[col] = (df_match[col].astype("string").str.strip().str.lower().fillna("__missing__"))
    return df_match


def compute_record_linkage_risk(container_input_real, container_input_synth, container_output, quasi_identifier_columns):
    """
    Record Linkage Risk Evaluation using exact matching.

    This component estimates linkage risk by comparing quasi-identifier
    patterns in synthetic data against quasi-identifier patterns in real data.

    A synthetic record is considered:
      - unmatched: no real record has the same QI pattern
      - uniquely matched: exactly one real record has the same QI pattern
      - ambiguously matched: multiple real records have the same QI pattern

    Unique matches indicate higher identity disclosure risk.
    """

    df_real = pd.read_parquet(container_input_real)
    df_synth = pd.read_parquet(container_input_synth)

    missing_real = [
        col for col in quasi_identifier_columns
        if col not in df_real.columns
    ]

    missing_synth = [
        col for col in quasi_identifier_columns
        if col not in df_synth.columns
    ]

    if missing_real:
        raise ValueError(f"Quasi-identifier columns missing in real dataset: {missing_real}")

    if missing_synth:
        raise ValueError(f"Quasi-identifier columns missing in synthetic dataset: {missing_synth}")

    real_qi = normalize_quasi_identifier_values(df_real, quasi_identifier_columns)

    synth_qi = normalize_quasi_identifier_values(df_synth, quasi_identifier_columns)

    # Count equivalence classes in the real dataset
    real_equivalence_classes = (real_qi.groupby(quasi_identifier_columns).size().reset_index(name="real_match_count"))

    # Count equivalence classes in the synthetic dataset
    synth_equivalence_classes = (synth_qi.groupby(quasi_identifier_columns).size().reset_index(name="synthetic_class_count"))

    # Match synthetic QI patterns against real QI patterns
    matched = synth_qi.merge(real_equivalence_classes, on=quasi_identifier_columns, how="left")

    matched["real_match_count"] = (matched["real_match_count"].fillna(0).astype(int))

    unmatched_count = int((matched["real_match_count"] == 0).sum())

    unique_match_count = int((matched["real_match_count"] == 1).sum())

    ambiguous_match_count = int((matched["real_match_count"] > 1).sum())

    total_synthetic_records = len(matched)

    unique_match_rate = (
        unique_match_count / total_synthetic_records
        if total_synthetic_records > 0
        else 0.0
    )

    ambiguous_match_rate = (
        ambiguous_match_count / total_synthetic_records
        if total_synthetic_records > 0
        else 0.0
    )

    unmatched_rate = (
        unmatched_count / total_synthetic_records
        if total_synthetic_records > 0
        else 0.0
    )

    # Linkage probability:
    # If a synthetic record matches n real records, probability of correct
    # random linkage is 1/n. If n=0, probability is 0.
    matched["linkage_probability"] = matched["real_match_count"].apply(
        lambda n: 1.0 / n if n > 0 else 0.0
    )

    mean_linkage_probability = float(
        matched["linkage_probability"].mean()
    )

    max_linkage_probability = float(
        matched["linkage_probability"].max()
    )

    # Risk interpretation based on unique match rate
    if unique_match_rate < 0.10:
        risk_level = "LOW"
    elif unique_match_rate < 0.30:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    summary = pd.DataFrame([{
        "metric_name": "Record Linkage Risk",
        "method": "Exact matching on user-specified quasi-identifiers",
        "quasi_identifiers": ",".join(quasi_identifier_columns),
        "real_records": len(df_real),
        "synthetic_records": len(df_synth),
        "unmatched_count": unmatched_count,
        "unique_match_count": unique_match_count,
        "ambiguous_match_count": ambiguous_match_count,
        "unmatched_rate": round(unmatched_rate, 4),
        "unique_match_rate": round(unique_match_rate, 4),
        "ambiguous_match_rate": round(ambiguous_match_rate, 4),
        "mean_linkage_probability": round(mean_linkage_probability, 4),
        "max_linkage_probability": round(max_linkage_probability, 4),
        "real_equivalence_class_count": len(real_equivalence_classes),
        "synthetic_equivalence_class_count": len(synth_equivalence_classes),
        "risk_level": risk_level,
        "interpretation": (
            "Unique matches indicate higher record linkage risk because a "
            "synthetic quasi-identifier pattern corresponds to exactly one "
            "real record. Ambiguous matches are less risky because multiple "
            "real records share the same quasi-identifier pattern."
        )
    }])

    summary.to_parquet(container_output, index=False)

    print("✅ Record Linkage Risk evaluation completed successfully.")
    print(summary)


if __name__ == "__main__":
    container_input_real = os.environ.get("container_input_real")
    container_input_synth = os.environ.get("container_input_synth")
    container_output = os.environ.get("container_output")
    quasi_identifier_columns = json.loads(os.environ.get("quasi_identifier_columns", "[]"))

    if not container_input_real:
        raise ValueError("container_input_real is required.")

    if not container_input_synth:
        raise ValueError("container_input_synth is required.")

    if not container_output:
        raise ValueError("container_output is required.")

    if not quasi_identifier_columns:
        raise ValueError("quasi_identifier_columns is required.")

    compute_record_linkage_risk(
        container_input_real=container_input_real,
        container_input_synth=container_input_synth,
        container_output=container_output,
        quasi_identifier_columns=quasi_identifier_columns
    )
