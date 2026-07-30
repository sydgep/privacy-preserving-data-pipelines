import os
import json
import pandas as pd


def compute_l_diversity(container_input, container_output, quasi_identifier_columns, sensitive_attribute, expected_l):
    """
    Evaluate whether a dataset satisfies a user-specified distinct
    l-diversity requirement for the selected sensitive attribute.

    The l-diversity requirement is satisfied when every equivalence class
    formed by the quasi-identifiers contains at least expected_l distinct
    values of the sensitive attribute.
    """

    if not container_input:
        raise ValueError("container_input is required.")

    if not container_output:
        raise ValueError("container_output is required.")

    if not quasi_identifier_columns:
        raise ValueError("quasi_identifier_columns must not be empty.")

    if not sensitive_attribute:
        raise ValueError("sensitive_attribute is required.")

    if expected_l < 1:
        raise ValueError("expected_l must be an integer greater than or equal to 1.")

    df = pd.read_parquet(container_input)

    if df.empty:
        raise ValueError("The input dataset is empty. l-diversity cannot be evaluated.")

    missing_qi_columns = [column for column in quasi_identifier_columns  if column not in df.columns]

    if missing_qi_columns:
        raise ValueError(f"Quasi-identifier columns not found: {missing_qi_columns}")

    if sensitive_attribute not in df.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found.")

    qi_sensitive_df = df[quasi_identifier_columns + [sensitive_attribute]].copy()

    # Treat missing values as explicit categories.
    qi_sensitive_df = qi_sensitive_df.fillna("__missing__")

    diversity_table = (qi_sensitive_df.groupby(quasi_identifier_columns, dropna=False)[sensitive_attribute].nunique().reset_index(name="distinct_sensitive_values"))

    class_sizes = (qi_sensitive_df.groupby(quasi_identifier_columns, dropna=False).size().reset_index(name="equivalence_class_size"))

    diversity_table = diversity_table.merge(class_sizes, on=quasi_identifier_columns, how="left")

    total_records = int(len(df))
    equivalence_class_count = int(len(diversity_table))

    observed_minimum_l = int(diversity_table["distinct_sensitive_values"].min())

    maximum_distinct_sensitive_values = int(diversity_table["distinct_sensitive_values"].max())

    violating_classes = diversity_table[diversity_table["distinct_sensitive_values"] < expected_l]

    violating_class_count = int(len(violating_classes))

    violating_record_count = int(violating_classes["equivalence_class_size"].sum())

    violating_record_rate = (violating_record_count / total_records
        if total_records > 0
        else 0.0
    )

    status = ("Satisfied" if observed_minimum_l >= expected_l else "Not satisfied")

    if status == "Satisfied":
        interpretation = (
            f"The dataset satisfies the specified distinct l-diversity "
            f"requirement of l={expected_l} for the sensitive attribute "
            f"'{sensitive_attribute}'. The smallest number of distinct "
            f"sensitive values observed in any equivalence class is "
            f"{observed_minimum_l}. No equivalence class contains fewer "
            f"than {expected_l} distinct values of the sensitive attribute."
        )
    else:
        interpretation = (
            f"The dataset does not satisfy the specified distinct "
            f"l-diversity requirement of l={expected_l} for the sensitive "
            f"attribute '{sensitive_attribute}'. The smallest number of "
            f"distinct sensitive values observed in an equivalence class "
            f"is {observed_minimum_l}. "
            f"{violating_record_count} out of {total_records} records "
            f"({violating_record_rate * 100:.2f}%) belong to "
            f"{violating_class_count} equivalence classes containing fewer "
            f"than {expected_l} distinct values of the sensitive attribute."
        )

    summary = pd.DataFrame([{
        "metric_name": "l-Diversity",
        "status": status,
        "method": ("Distinct sensitive attribute values within quasi-identifier equivalence classes"),
        "quasi_identifiers": ",".join(quasi_identifier_columns),
        "sensitive_attribute": sensitive_attribute,
        "expected_l": int(expected_l),
        "observed_minimum_l": observed_minimum_l,
        "maximum_distinct_sensitive_values": (maximum_distinct_sensitive_values),
        "equivalence_class_count": equivalence_class_count,
        "violating_class_count": violating_class_count,
        "violating_record_count": violating_record_count,
        "total_records": total_records,
        "violating_record_rate": round(violating_record_rate, 6),
        "violating_record_percentage": round(violating_record_rate * 100, 2),
        "interpretation": interpretation,
    }])

    summary.to_parquet(container_output, index=False)

    print("✅ l-Diversity evaluation completed successfully.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    container_input = os.environ.get("container_input")
    container_output = os.environ.get("container_output")

    try:
        quasi_identifier_columns = json.loads(os.environ.get("quasi_identifier_columns", "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("quasi_identifier_columns must be a valid JSON list.") from exc

    sensitive_attribute = os.environ.get("sensitive_attribute")

    expected_l = int(os.environ.get("expected_l", "2"))

    compute_l_diversity(
        container_input=container_input,
        container_output=container_output,
        quasi_identifier_columns=quasi_identifier_columns,
        sensitive_attribute=sensitive_attribute,
        expected_l=expected_l
    )
