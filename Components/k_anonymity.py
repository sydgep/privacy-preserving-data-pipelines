import os
import json
import pandas as pd


def compute_k_anonymity(container_input, container_output, quasi_identifier_columns, expected_k):
    """
    Evaluate whether a dataset satisfies a user-specified k-anonymity
    requirement for the selected quasi-identifiers.

    The k-anonymity requirement is satisfied when every equivalence class
    formed by the quasi-identifiers contains at least expected_k records.
    """

    if not container_input:
        raise ValueError("container_input is required.")

    if not container_output:
        raise ValueError("container_output is required.")

    if not quasi_identifier_columns:
        raise ValueError("quasi_identifier_columns must not be empty.")

    if expected_k < 1:
        raise ValueError("expected_k must be an integer greater than or equal to 1.")

    df = pd.read_parquet(container_input)

    if df.empty:
        raise ValueError("The input dataset is empty. k-anonymity cannot be evaluated.")

    missing_columns = [column for column in quasi_identifier_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Quasi-identifier columns not found: {missing_columns}")

    qi_df = df[quasi_identifier_columns].copy()

    # Treat missing quasi-identifier values as an explicit category.
    qi_df = qi_df.fillna("__missing__")

    equivalence_classes = (qi_df.groupby(quasi_identifier_columns, dropna=False).size().reset_index(name="equivalence_class_size"))

    total_records = int(len(df))
    equivalence_class_count = int(len(equivalence_classes))

    observed_minimum_k = int(equivalence_classes["equivalence_class_size"].min())

    maximum_equivalence_class_size = int(equivalence_classes["equivalence_class_size"].max())

    violating_classes = equivalence_classes[equivalence_classes["equivalence_class_size"] < expected_k]

    violating_class_count = int(len(violating_classes))

    violating_record_count = int(violating_classes["equivalence_class_size"].sum())

    violating_record_rate = (violating_record_count / total_records
        if total_records > 0
        else 0.0
    )

    status = ("Satisfied" if observed_minimum_k >= expected_k else "Not satisfied")

    if status == "Satisfied":
        interpretation = (
            f"The dataset satisfies the specified k-anonymity requirement "
            f"of k={expected_k} for the selected quasi-identifiers. "
            f"The smallest equivalence class contains "
            f"{observed_minimum_k} records. No equivalence class contains "
            f"fewer than {expected_k} records."
        )
    else:
        interpretation = (
            f"The dataset does not satisfy the specified k-anonymity "
            f"requirement of k={expected_k} for the selected "
            f"quasi-identifiers. The smallest equivalence class contains "
            f"{observed_minimum_k} records. "
            f"{violating_record_count} out of {total_records} records "
            f"({violating_record_rate * 100:.2f}%) belong to "
            f"{violating_class_count} equivalence classes containing fewer "
            f"than {expected_k} records."
        )

    summary = pd.DataFrame([{
        "metric_name": "k-Anonymity",
        "status": status,
        "method": ("Equivalence class size over user-specified quasi-identifiers"),
        "quasi_identifiers": ",".join(quasi_identifier_columns),
        "expected_k": int(expected_k),
        "observed_minimum_k": observed_minimum_k,
        "maximum_equivalence_class_size": (maximum_equivalence_class_size),
        "equivalence_class_count": equivalence_class_count,
        "violating_class_count": violating_class_count,
        "violating_record_count": violating_record_count,
        "total_records": total_records,
        "violating_record_rate": round(violating_record_rate, 6),
        "violating_record_percentage": round(violating_record_rate * 100, 2),
        "interpretation": interpretation,
    }])

    summary.to_parquet(container_output, index=False)

    print("✅ k-Anonymity evaluation completed successfully.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    container_input = os.environ.get("container_input")
    container_output = os.environ.get("container_output")

    try:
        quasi_identifier_columns = json.loads(os.environ.get("quasi_identifier_columns", "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("quasi_identifier_columns must be a valid JSON list.") from exc

    expected_k = int(os.environ.get("expected_k", "5"))

    compute_k_anonymity(
        container_input=container_input,
        container_output=container_output,
        quasi_identifier_columns=quasi_identifier_columns,
        expected_k=expected_k
    )
