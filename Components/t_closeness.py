import os
import json
import pandas as pd
import numpy as np


def total_variation_distance(
    group_distribution,
    global_distribution
):
    """
    Compute the Total Variation Distance between two probability
    distributions.

    TVD = 0.5 * sum(|P_group(x) - P_global(x)|)

    Value range:
        0 = identical distributions
        1 = completely different distributions
    """

    all_values = global_distribution.index

    group_distribution = group_distribution.reindex(
        all_values,
        fill_value=0
    )

    distance = 0.5 * np.abs(
        group_distribution - global_distribution
    ).sum()

    return float(distance)


def compute_t_closeness(
    container_input,
    container_output,
    quasi_identifier_columns,
    sensitive_attribute,
    expected_t
):
    """
    Evaluate whether a dataset satisfies a user-specified t-closeness
    requirement for the selected sensitive attribute.

    The t-closeness requirement is satisfied when the Total Variation
    Distance between the sensitive-attribute distribution of every
    equivalence class and the global sensitive-attribute distribution
    is less than or equal to expected_t.
    """

    if not container_input:
        raise ValueError("container_input is required.")

    if not container_output:
        raise ValueError("container_output is required.")

    if not quasi_identifier_columns:
        raise ValueError(
            "quasi_identifier_columns must not be empty."
        )

    if not sensitive_attribute:
        raise ValueError(
            "sensitive_attribute is required."
        )

    if not 0 <= expected_t <= 1:
        raise ValueError(
            "expected_t must be between 0 and 1."
        )

    df = pd.read_parquet(container_input)

    if df.empty:
        raise ValueError(
            "The input dataset is empty. t-closeness cannot be evaluated."
        )

    missing_qi_columns = [
        column
        for column in quasi_identifier_columns
        if column not in df.columns
    ]

    if missing_qi_columns:
        raise ValueError(
            f"Quasi-identifier columns not found: {missing_qi_columns}"
        )

    if sensitive_attribute not in df.columns:
        raise ValueError(
            f"Sensitive attribute '{sensitive_attribute}' not found."
        )

    working_df = df[
        quasi_identifier_columns + [sensitive_attribute]
    ].copy()

    # Treat missing values as explicit categories.
    working_df = working_df.fillna("__missing__")

    global_distribution = (
        working_df[sensitive_attribute]
        .value_counts(normalize=True)
        .sort_index()
    )

    group_results = []

    grouped = working_df.groupby(
        quasi_identifier_columns,
        dropna=False
    )

    for group_key, group_df in grouped:
        group_distribution = (
            group_df[sensitive_attribute]
            .value_counts(normalize=True)
            .sort_index()
        )

        distance = total_variation_distance(
            group_distribution,
            global_distribution
        )

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_record = {
            column: value
            for column, value in zip(
                quasi_identifier_columns,
                group_key
            )
        }

        group_record["equivalence_class_size"] = int(
            len(group_df)
        )

        group_record["t_distance"] = distance

        group_results.append(group_record)

    closeness_table = pd.DataFrame(group_results)

    total_records = int(len(working_df))
    equivalence_class_count = int(len(closeness_table))

    observed_maximum_t = float(
        closeness_table["t_distance"].max()
    )

    observed_minimum_t = float(
        closeness_table["t_distance"].min()
    )

    violating_classes = closeness_table[
        closeness_table["t_distance"] > expected_t
    ]

    violating_class_count = int(
        len(violating_classes)
    )

    violating_record_count = int(
        violating_classes["equivalence_class_size"].sum()
    )

    violating_record_rate = (
        violating_record_count / total_records
        if total_records > 0
        else 0.0
    )

    status = (
        "Satisfied"
        if observed_maximum_t <= expected_t
        else "Not satisfied"
    )

    if status == "Satisfied":
        interpretation = (
            f"The dataset satisfies the specified t-closeness "
            f"requirement of t={expected_t} for the sensitive attribute "
            f"'{sensitive_attribute}'. The largest Total Variation "
            f"Distance observed between an equivalence-class distribution "
            f"and the global sensitive-attribute distribution is "
            f"{observed_maximum_t:.6f}. No equivalence class has a "
            f"t-distance greater than {expected_t}."
        )
    else:
        interpretation = (
            f"The dataset does not satisfy the specified t-closeness "
            f"requirement of t={expected_t} for the sensitive attribute "
            f"'{sensitive_attribute}'. The largest Total Variation "
            f"Distance observed between an equivalence-class distribution "
            f"and the global sensitive-attribute distribution is "
            f"{observed_maximum_t:.6f}. "
            f"{violating_record_count} out of {total_records} records "
            f"({violating_record_rate * 100:.2f}%) belong to "
            f"{violating_class_count} equivalence classes with a "
            f"t-distance greater than {expected_t}."
        )

    summary = pd.DataFrame([{
        "metric_name": "t-Closeness",
        "status": status,
        "method": (
            "Total Variation Distance between each equivalence-class "
            "sensitive-attribute distribution and the global "
            "sensitive-attribute distribution"
        ),
        "quasi_identifiers": ",".join(
            quasi_identifier_columns
        ),
        "sensitive_attribute": sensitive_attribute,
        "expected_t": float(expected_t),
        "observed_maximum_t": round(
            observed_maximum_t,
            6
        ),
        "observed_minimum_t": round(
            observed_minimum_t,
            6
        ),
        "equivalence_class_count": equivalence_class_count,
        "violating_class_count": violating_class_count,
        "violating_record_count": violating_record_count,
        "total_records": total_records,
        "violating_record_rate": round(
            violating_record_rate,
            6
        ),
        "violating_record_percentage": round(
            violating_record_rate * 100,
            2
        ),
        "interpretation": interpretation,
    }])

    summary.to_parquet(
        container_output,
        index=False
    )

    print("✅ t-Closeness evaluation completed successfully.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    container_input = os.environ.get("container_input")
    container_output = os.environ.get("container_output")

    try:
        quasi_identifier_columns = json.loads(
            os.environ.get(
                "quasi_identifier_columns",
                "[]"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "quasi_identifier_columns must be a valid JSON list."
        ) from exc

    sensitive_attribute = os.environ.get(
        "sensitive_attribute"
    )

    expected_t = float(
        os.environ.get(
            "expected_t",
            "0.2"
        )
    )

    compute_t_closeness(
        container_input=container_input,
        container_output=container_output,
        quasi_identifier_columns=quasi_identifier_columns,
        sensitive_attribute=sensitive_attribute,
        expected_t=expected_t
    )
