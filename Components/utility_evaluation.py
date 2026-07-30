import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


# ------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------
RANDOM_STATE = 42

CLASSIFICATION_METRICS = {"accuracy", "f1", "roc_auc"}
REGRESSION_METRICS = {"r2", "rmse"}


# ------------------------------------------------------------------
# Supported model registry
# ------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "LogisticRegression": {
        "type": "classification",
        "factory": lambda: LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    },
    "RandomForestClassifier": {
        "type": "classification",
        "factory": lambda: RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    },
    "GradientBoostingClassifier": {
        "type": "classification",
        "factory": lambda: GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    },
    "SVC": {
        "type": "classification",
        "factory": lambda: SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
    },
    "LinearRegression": {
        "type": "regression",
        "factory": lambda: LinearRegression(),
    },
    "RandomForestRegressor": {
        "type": "regression",
        "factory": lambda: RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    },
    "GradientBoostingRegressor": {
        "type": "regression",
        "factory": lambda: GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
    },
    "SVR": {
        "type": "regression",
        "factory": lambda: SVR(),
    },
}


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def parse_list(value: Optional[str]) -> List[str]:
    """
    Parse either:
      - a JSON list, e.g. '["Age", "Gender"]'
      - a comma-separated string, e.g. 'Age,Gender'
    """
    if value is None or not value.strip():
        return []

    value = value.strip()

    if value.startswith("["):
        parsed = json.loads(value)

        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON list.")

        return [
            str(item).strip()
            for item in parsed
            if str(item).strip()
        ]

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def load_parquet(path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load and validate a Parquet dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{dataset_name} dataset not found: {path}"
        )

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(
            f"{dataset_name} dataset is empty."
        )

    if df.columns.duplicated().any():
        duplicated_columns = (
            df.columns[df.columns.duplicated()]
            .tolist()
        )

        raise ValueError(
            f"{dataset_name} dataset contains duplicate columns: "
            f"{duplicated_columns}"
        )

    return df.replace([np.inf, -np.inf], np.nan)


def validate_parameters(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    feature_columns: Sequence[str],
    task_type: str,
    train_size: float,
) -> None:
    """
    Validate datasets and evaluation parameters.
    """
    if not 0.0 < train_size < 1.0:
        raise ValueError(
            "train_size must be greater than 0 and less than 1."
        )

    if not feature_columns:
        raise ValueError(
            "feature_columns must contain at least one column."
        )

    if target_col in feature_columns:
        raise ValueError(
            "target_col must not be included in feature_columns."
        )

    required_columns = set(feature_columns) | {target_col}

    for dataset_name, df in (
        ("Real", real_df),
        ("Synthetic", synth_df),
    ):
        missing_columns = sorted(
            required_columns - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{dataset_name} dataset is missing required columns: "
                f"{missing_columns}"
            )

        missing_targets = int(
            df[target_col].isna().sum()
        )

        if missing_targets > 0:
            raise ValueError(
                f"{dataset_name} dataset contains "
                f"{missing_targets} missing values in target column "
                f"'{target_col}'."
            )

    if task_type == "classification":
        real_classes = real_df[target_col].nunique(dropna=True)
        synth_classes = synth_df[target_col].nunique(dropna=True)

        if real_classes < 2:
            raise ValueError(
                "The real target column must contain at least two classes."
            )

        if synth_classes < 2:
            raise ValueError(
                "The synthetic target column must contain at least two classes."
            )

    elif task_type == "regression":
        for dataset_name, df in (
            ("Real", real_df),
            ("Synthetic", synth_df),
        ):
            numeric_target = pd.to_numeric(
                df[target_col],
                errors="coerce",
            )

            if numeric_target.isna().any():
                raise ValueError(
                    f"{dataset_name} regression target '{target_col}' "
                    "contains non-numeric values."
                )


def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    train_size: float,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Split a dataset into training and test sets.

    Classification datasets are stratified where possible.
    """
    stratify_values = (
        df[target_col]
        if task_type == "classification"
        else None
    )

    split_note = None

    if stratify_values is not None:
        class_counts = stratify_values.value_counts(
            dropna=False
        )

        if (class_counts < 2).any():
            rare_classes = (
                class_counts[class_counts < 2]
                .index.astype(str)
                .tolist()
            )

            stratify_values = None

            split_note = (
                f"{dataset_name} split was not stratified because "
                f"these classes contain fewer than two records: "
                f"{rare_classes}"
            )

    try:
        train_df, test_df = train_test_split(
            df,
            train_size=train_size,
            random_state=RANDOM_STATE,
            stratify=stratify_values,
        )

    except ValueError as exc:
        if stratify_values is None:
            raise

        train_df, test_df = train_test_split(
            df,
            train_size=train_size,
            random_state=RANDOM_STATE,
            stratify=None,
        )

        split_note = (
            f"{dataset_name} split fell back to an unstratified "
            f"split: {exc}"
        )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_note,
    )


def infer_feature_types(
    training_features: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Detect numerical and categorical columns from the training dataset.
    """
    numeric_columns = training_features.select_dtypes(
        include=[np.number, "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in training_features.columns
        if column not in numeric_columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError(
            "No usable feature columns were detected."
        )

    return numeric_columns, categorical_columns


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a version-compatible OneHotEncoder.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipelines for numerical and categorical data.
    """
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                list(numeric_columns),
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__missing__",
                    ),
                ),
                (
                    "encoder",
                    make_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(categorical_columns),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def coerce_feature_types(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coerce test features to the feature types inferred from training data.
    """
    train = train_features.copy()
    test = test_features.copy()

    for column in numeric_columns:
        train[column] = pd.to_numeric(
            train[column],
            errors="coerce",
        )

        test[column] = pd.to_numeric(
            test[column],
            errors="coerce",
        )

    for column in categorical_columns:
        train[column] = (
            train[column]
            .where(train[column].notna(), np.nan)
            .astype(object)
        )

        test[column] = (
            test[column]
            .where(test[column].notna(), np.nan)
            .astype(object)
        )

    return train, test


# ------------------------------------------------------------------
# Metric calculation
# ------------------------------------------------------------------
def calculate_classification_metrics(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    metric_names: Sequence[str],
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """
    Calculate requested classification metrics.
    """
    predictions = pipeline.predict(x_test)

    scores: Dict[str, Optional[float]] = {}
    notes: List[str] = []

    if "accuracy" in metric_names:
        scores["accuracy_score"] = float(
            accuracy_score(y_test, predictions)
        )

    if "f1" in metric_names:
        scores["f1_score"] = float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

    if "roc_auc" in metric_names:
        estimator = pipeline.named_steps["model"]
        model_classes = np.asarray(estimator.classes_)

        test_classes = set(
            pd.Series(y_test).unique()
        )

        training_classes = set(
            model_classes.tolist()
        )

        unseen_test_classes = sorted(
            str(value)
            for value in test_classes - training_classes
        )

        if unseen_test_classes:
            scores["roc_auc_score"] = None

            notes.append(
                "ROC-AUC was not calculated because the test data "
                "contain classes absent from the training data: "
                f"{unseen_test_classes}"
            )

        elif pd.Series(y_test).nunique() < 2:
            scores["roc_auc_score"] = None

            notes.append(
                "ROC-AUC was not calculated because the test target "
                "contains fewer than two classes."
            )

        else:
            try:
                if hasattr(pipeline, "predict_proba"):
                    probability_scores = pipeline.predict_proba(
                        x_test
                    )

                elif hasattr(pipeline, "decision_function"):
                    probability_scores = pipeline.decision_function(
                        x_test
                    )

                else:
                    probability_scores = None

                if probability_scores is None:
                    scores["roc_auc_score"] = None

                    notes.append(
                        "ROC-AUC was not calculated because the model "
                        "provides neither predict_proba nor "
                        "decision_function."
                    )

                elif len(model_classes) == 2:
                    positive_class = model_classes[1]

                    binary_target = (
                        pd.Series(y_test).to_numpy()
                        == positive_class
                    ).astype(int)

                    if np.ndim(probability_scores) == 2:
                        auc_input = probability_scores[:, 1]
                    else:
                        auc_input = probability_scores

                    scores["roc_auc_score"] = float(
                        roc_auc_score(
                            binary_target,
                            auc_input,
                        )
                    )

                else:
                    scores["roc_auc_score"] = float(
                        roc_auc_score(
                            y_test,
                            probability_scores,
                            labels=model_classes,
                            multi_class="ovr",
                            average="macro",
                        )
                    )

            except Exception as exc:
                scores["roc_auc_score"] = None

                notes.append(
                    f"ROC-AUC could not be calculated: {exc}"
                )

    return scores, notes


def calculate_regression_metrics(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    metric_names: Sequence[str],
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """
    Calculate requested regression metrics.
    """
    predictions = pipeline.predict(x_test)

    scores: Dict[str, Optional[float]] = {}

    if "r2" in metric_names:
        scores["r2_score"] = float(
            r2_score(y_test, predictions)
        )

    if "rmse" in metric_names:
        scores["rmse_score"] = float(
            mean_squared_error(
                y_test,
                predictions,
            ) ** 0.5
        )

    return scores, []

# ------------------------------------------------------------------
# ML experiment evaluation
# ------------------------------------------------------------------
def evaluate_mode(
    ml_experiment: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_columns: Sequence[str],
    model_name: str,
    model_factory: Callable[[], object],
    task_type: str,
    metric_names: Sequence[str],
    initial_notes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    Train and evaluate one machine-learning experiment:

      - TrainReal_TestReal:
        Train on real data and test on held-out real data.

      - TrainSynth_TestReal:
        Train on synthetic data and test on held-out real data.
    """
    selected_columns = list(feature_columns) + [target_col]

    train_working = train_df[selected_columns].copy()
    test_working = test_df[selected_columns].copy()

    y_train = train_working[target_col]
    y_test = test_working[target_col]

    x_train = train_working.drop(columns=[target_col])
    x_test = test_working.drop(columns=[target_col])

    if list(x_train.columns) != list(x_test.columns):
        raise ValueError(
            f"Feature schema mismatch in experiment "
            f"'{ml_experiment}'. "
            f"Training columns: {list(x_train.columns)}. "
            f"Test columns: {list(x_test.columns)}."
        )

    numeric_columns, categorical_columns = infer_feature_types(
        x_train
    )

    x_train, x_test = coerce_feature_types(
        train_features=x_train,
        test_features=x_test,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    if task_type == "regression":
        y_train = pd.to_numeric(
            y_train,
            errors="raise",
        )

        y_test = pd.to_numeric(
            y_test,
            errors="raise",
        )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                ),
            ),
            (
                "model",
                model_factory(),
            ),
        ]
    )

    pipeline.fit(
        x_train,
        y_train,
    )

    if task_type == "classification":
        scores, metric_notes = calculate_classification_metrics(
            pipeline=pipeline,
            x_test=x_test,
            y_test=y_test,
            metric_names=metric_names,
        )

    else:
        scores, metric_notes = calculate_regression_metrics(
            pipeline=pipeline,
            x_test=x_test,
            y_test=y_test,
            metric_names=metric_names,
        )

    notes = list(initial_notes or []) + metric_notes

    rounded_scores = {
        metric_name: (
            round(metric_value, 4)
            if metric_value is not None
            else None
        )
        for metric_name, metric_value in scores.items()
    }

    return {
        "ml_experiment": ml_experiment,
        "training_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "model": model_name,
        **rounded_scores,
        "notes": (
            " | ".join(notes)
            if notes
            else None
        ),
    }


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------
def utility_evaluation(
    container_input_real: str,
    container_input_synth: str,
    target_col: str,
    feature_columns: Sequence[str],
    train_size: float,
    model_name: str,
    metrics: str,
    container_output: str,
) -> None:
    """
    Evaluate synthetic-data utility using two ML experiments:

      TrainReal_TestReal:
        Train on real training data.
        Test on held-out real data.

      TrainSynth_TestReal:
        Train on synthetic training data.
        Test on the same held-out real data.
    """
    real_df = load_parquet(
        container_input_real,
        "Real",
    )

    synth_df = load_parquet(
        container_input_synth,
        "Synthetic",
    )

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models are: "
            f"{sorted(MODEL_REGISTRY.keys())}"
        )

    model_spec = MODEL_REGISTRY[model_name]

    task_type = str(model_spec["type"])
    model_factory = model_spec["factory"]

    selected_features = list(
        dict.fromkeys(feature_columns)
    )

    validate_parameters(
        real_df=real_df,
        synth_df=synth_df,
        target_col=target_col,
        feature_columns=selected_features,
        task_type=task_type,
        train_size=train_size,
    )

    metric_names = parse_list(metrics)

    allowed_metrics = (
        CLASSIFICATION_METRICS
        if task_type == "classification"
        else REGRESSION_METRICS
    )

    invalid_metrics = [
        metric
        for metric in metric_names
        if metric not in allowed_metrics
    ]

    if invalid_metrics:
        raise ValueError(
            f"Invalid metrics for {task_type}: "
            f"{invalid_metrics}. "
            f"Allowed metrics are: "
            f"{sorted(allowed_metrics)}"
        )

    real_train, real_test, real_split_note = split_dataset(
        df=real_df,
        target_col=target_col,
        task_type=task_type,
        train_size=train_size,
        dataset_name="Real",
    )

    synth_train, _, synth_split_note = split_dataset(
        df=synth_df,
        target_col=target_col,
        task_type=task_type,
        train_size=train_size,
        dataset_name="Synthetic",
    )

    ml_experiments = [
        (
            "TrainReal_TestReal",
            real_train,
            real_test,
            [
                note
                for note in [real_split_note]
                if note
            ],
        ),
        (
            "TrainSynth_TestReal",
            synth_train,
            real_test,
            [
                note
                for note in [
                    synth_split_note,
                    real_split_note,
                ]
                if note
            ],
        ),
    ]

    results = []

    for (
        ml_experiment,
        train_df,
        test_df,
        notes,
    ) in ml_experiments:
        print(
            f"Evaluating ML experiment: "
            f"{ml_experiment}"
        )

        result = evaluate_mode(
            ml_experiment=ml_experiment,
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            feature_columns=selected_features,
            model_name=model_name,
            model_factory=model_factory,
            task_type=task_type,
            metric_names=metric_names,
            initial_notes=notes,
        )

        results.append(result)

    result_df = pd.DataFrame(results)

    if task_type == "classification":
        metric_columns = [
            "accuracy_score",
            "f1_score",
            "roc_auc_score",
        ]
    else:
        metric_columns = [
            "r2_score",
            "rmse_score",
        ]

    output_columns = [
        "ml_experiment",
        "training_records",
        "test_records",
        "model",
        *[
            column
            for column in metric_columns
            if column in result_df.columns
        ],
        "notes",
    ]

    result_df = result_df[output_columns]

    output_directory = os.path.dirname(
        container_output
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    result_df.to_parquet(
        container_output,
        index=False,
    )

    print(
        f"Utility evaluation saved to: "
        f"{container_output}"
    )
    print(f"Task type: {task_type}")
    print(f"Model: {model_name}")
    print(f"Target column: {target_col}")
    print(f"Feature columns: {selected_features}")
    print(f"Random state: {RANDOM_STATE}")
    print(result_df.to_string(index=False))

# ------------------------------------------------------------------
# Container entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    container_input_real = os.environ.get("container_input_real")
    container_input_synth = os.environ.get("container_input_synth")
    container_output = os.environ.get("container_output")
    target_col = os.environ.get("target_col")
    feature_columns = parse_list(os.environ.get("feature_columns"))
    model_name = os.environ.get("model")
    train_size = float(os.environ.get("train_size", "0.7",))
    metrics = os.environ.get("metrics", "accuracy,f1,roc_auc",)

    utility_evaluation(
        container_input_real=container_input_real,
        container_input_synth=container_input_synth,
        target_col=target_col,
        feature_columns=feature_columns,
        train_size=train_size,
        model_name=model_name,
        metrics=metrics,
        container_output=container_output,
    )
