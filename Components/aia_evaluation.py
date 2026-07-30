import os
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def create_attack_model(model_name):
    model_name = model_name.lower().strip()

    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=500)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

    raise ValueError("Unsupported model. Choose one of logistic_regression, random_forest, gradient_boosting" )


def run_attribute_inference_attack(container_input_real, container_input_synth, quasi_identifier_columns, sensitive_attribute, container_output, model_name="random_forest"):
    """
    Comparative Attribute Inference Attack (AIA)

    The attacker trains on the released synthetic dataset and attempts
    to infer sensitive attributes in the real dataset.

    If the model trained on synthetic data predicts real sensitive
    attributes accurately, this indicates attribute disclosure risk.
    """

    df_real = pd.read_parquet(container_input_real)
    df_synth = pd.read_parquet(container_input_synth)

    required_columns = quasi_identifier_columns + [sensitive_attribute]

    for column in required_columns:
        if column not in df_real.columns:
            raise ValueError(f"Column '{column}' not found in real dataset.")

        if column not in df_synth.columns:
            raise ValueError(f"Column '{column}' not found in synthetic dataset.")

    df_real = df_real[required_columns].dropna().copy()
    df_synth = df_synth[required_columns].dropna().copy()

    if df_real.empty:
        raise ValueError("Real dataset is empty after dropping missing values.")

    if df_synth.empty:
        raise ValueError("Synthetic dataset is empty after dropping missing values.")

    X_train = df_synth[quasi_identifier_columns].copy()
    y_train_raw = df_synth[sensitive_attribute].copy()

    X_test = df_real[quasi_identifier_columns].copy()
    y_test_raw = df_real[sensitive_attribute].copy()

    label_encoder = LabelEncoder()

    combined_target = pd.concat([y_train_raw, y_test_raw], ignore_index=True)

    label_encoder.fit(combined_target)

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    combined_features = pd.concat([X_train, X_test], ignore_index=True)

    categorical_columns = combined_features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numerical_columns = combined_features.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

    print("========== Attribute Inference Attack ==========")
    print("Training dataset: synthetic")
    print("Testing dataset : real")
    print(f"Attack model    : {model_name}")
    print(f"Sensitive attr. : {sensitive_attribute}")
    print(f"Quasi-identifiers: {quasi_identifier_columns}")
    print(f"Categorical QIs : {categorical_columns}")
    print(f"Numerical QIs   : {numerical_columns}")

    preprocessor = ColumnTransformer(transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns), ("numerical", "passthrough", numerical_columns)])

    classifier = create_attack_model(model_name)

    attack_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    attack_pipeline.fit(X_train, y_train)

    y_pred = attack_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    roc_auc = None

    if len(set(y_test)) == 2 and hasattr(attack_pipeline, "predict_proba"):
        y_pred_proba = attack_pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

    if accuracy < 0.60:
        risk_level = "LOW"
    elif accuracy < 0.80:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    summary = pd.DataFrame([{
        "metric_name": "Attribute Inference Attack (AIA)",
        "method": (
            "Attack model trained on synthetic data and tested on real data"
        ),
        "attack_model": model_name,
        "training_data": "synthetic",
        "testing_data": "real",
        "quasi_identifiers": ",".join(quasi_identifier_columns),
        "sensitive_attribute": sensitive_attribute,
        "categorical_quasi_identifiers": ",".join(categorical_columns),
        "numerical_quasi_identifiers": ",".join(numerical_columns),
        "real_records_used": len(df_real),
        "synthetic_records_used": len(df_synth),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "risk_level": risk_level,
        "interpretation": (
            "Higher predictive performance indicates greater attribute "
            "disclosure risk. This evaluates whether the released synthetic "
            "dataset preserves relationships that can be used to infer "
            "sensitive attributes in real records."
        )
    }])

    summary.to_parquet(container_output, index=False)

    print("✅ Attribute Inference Attack evaluation completed successfully.")
    print(summary)


if __name__ == "__main__":
    container_input_real = os.environ.get("container_input_real")
    container_input_synth = os.environ.get("container_input_synth")
    container_output = os.environ.get("container_output")

    quasi_identifier_columns = json.loads(os.environ.get("quasi_identifier_columns", "[]"))

    sensitive_attribute = os.environ.get("sensitive_attribute")

    model_name = os.environ.get("model", "random_forest")

    if not container_input_real:
        raise ValueError("container_input_real is required.")

    if not container_input_synth:
        raise ValueError("container_input_synth is required.")

    if not container_output:
        raise ValueError("container_output is required.")

    if not quasi_identifier_columns:
        raise ValueError("quasi_identifier_columns is required.")

    if not sensitive_attribute:
        raise ValueError("sensitive_attribute is required.")

    run_attribute_inference_attack(
        container_input_real=container_input_real,
        container_input_synth=container_input_synth,
        quasi_identifier_columns=quasi_identifier_columns,
        sensitive_attribute=sensitive_attribute,
        container_output=container_output,
        model_name=model_name
    )
