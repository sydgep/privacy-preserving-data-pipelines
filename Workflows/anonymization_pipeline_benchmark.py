import os
import json
import docker
import pandas as pd

from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact


# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
DATASET_DIR = "/path/to/Datasets"
HOST_SHARED_DIR = "/path/to/Datasets/outputs/bm_anonym"
os.makedirs(HOST_SHARED_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Container paths and volume mapping
# -----------------------------------------------------------------------------
CONTAINER_DATASET_DIR = "/app/datasets"
CONTAINER_OUTPUT_DIR = "/app/outputs"

VOLUME_MAPPING = {
    os.path.abspath(DATASET_DIR): {
        "bind": CONTAINER_DATASET_DIR,
        "mode": "ro",
    },
    os.path.abspath(HOST_SHARED_DIR): {
        "bind": CONTAINER_OUTPUT_DIR,
        "mode": "rw",
    },
}

docker_client = docker.from_env()


def create_parquet_artifact(file_name, key, description, n=20):
    """Create a Prefect table artifact from a Parquet output."""
    host_output = os.path.join(HOST_SHARED_DIR, file_name)

    if not os.path.exists(host_output):
        create_table_artifact(
            key=key,
            table=[{"error": f"Output file not found: {host_output}"}],
            description=description,
        )
        return

    try:
        df = pd.read_parquet(host_output)
        create_table_artifact(
            key=key,
            table=df.head(n).to_dict(orient="records"),
            description=description,
        )
    except Exception as exc:
        create_table_artifact(
            key=key,
            table=[{"error": str(exc)}],
            description=description,
        )


def run_container(image, environment, logger):
    """Run a task container and fail the Prefect task on container errors."""
    container = docker_client.containers.run(
        image=image,
        detach=True,
        volumes=VOLUME_MAPPING,
        environment=environment,
    )

    result = container.wait()
    logs = container.logs().decode(errors="replace")
    logger.info(logs)

    status_code = result.get("StatusCode", 1)
    container.remove()

    if status_code != 0:
        raise RuntimeError(
            f"Container '{image}' failed with exit code {status_code}.\n{logs}"
        )


@task(name="Data Ingestion")
def file_extraction(input_file, output_file, columns):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/file_extraction:latest",
        environment={
            "container_input": f"{CONTAINER_DATASET_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "columns": columns,
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "sydgep-benchmark-ingested-data",
        f"SYDGEP medical benchmark dataset ingested from {input_file}",
    )


@task(name="Direct Identifier Removal")
def direct_identifier_removal(input_file, output_file, identifier_columns):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/direct_identifier_removal:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "identifier_columns": json.dumps(identifier_columns),
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "de-identified-data",
        "De-identified dataset after DI removal",
    )


@task(name="Anonymization")
def anonymization(
    input_file,
    output_file,
    column,
    strategy,
    interval=None,
    precision=None,
):
    logger = get_run_logger()

    environment = {
        "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
        "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        "column": column,
        "strategy": strategy,
    }

    if interval is not None:
        environment["interval"] = str(interval)

    if precision is not None:
        environment["precision"] = precision

    run_container(
        image="ghcr.io/sydgep/docker-images/anonymization:latest",
        environment=environment,
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "anonymized-data",
        f"Dataset after anonymizing {column}",
    )


@task(name="k-Anonymity")
def k_anonymity(
    input_file,
    output_file,
    quasi_identifier_columns,
    expected_k,
):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/k_anonymity:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "quasi_identifier_columns": json.dumps(quasi_identifier_columns),
            "expected_k": str(expected_k),
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "k-anonymity-results",
        "k-anonymity evaluation results",
    )


@task(name="l-Diversity")
def l_diversity(
    input_file,
    output_file,
    quasi_identifier_columns,
    sensitive_attribute,
    expected_l,
):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/l_diversity:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "quasi_identifier_columns": json.dumps(quasi_identifier_columns),
            "sensitive_attribute": sensitive_attribute,
            "expected_l": str(expected_l),
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "l-diversity-results",
        "l-diversity evaluation results",
    )


@task(name="t-Closeness")
def t_closeness(
    input_file,
    output_file,
    quasi_identifier_columns,
    sensitive_attribute,
    expected_t,
):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/t_closeness:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "quasi_identifier_columns": json.dumps(quasi_identifier_columns),
            "sensitive_attribute": sensitive_attribute,
            "expected_t": str(expected_t),
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "t-closeness-results",
        "t-closeness evaluation results",
    )

@task(name="Statistical Evaluation")
def statistical_evaluation(real_data_file, synthetic_data_file, output_file):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/statistical_evaluation:v1",
        environment={
            "container_input_real": f"{CONTAINER_OUTPUT_DIR}/{real_data_file}",
            "container_input_synth": f"{CONTAINER_OUTPUT_DIR}/{synthetic_data_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "statistical-evaluation-results",
        "Statistical Evaluation results",
    )

@task(name="Utility Evaluation")
def utility_evaluation(real_data_file, synthetic_data_file, feature_columns, target_col, train_size, model_name, metrics, output_file): 
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/utility_evaluation:v1",
        environment={
            "container_input_real": f"{CONTAINER_OUTPUT_DIR}/{real_data_file}",
            "container_input_synth": f"{CONTAINER_OUTPUT_DIR}/{synthetic_data_file}",
            "feature_columns": json.dumps(feature_columns),
            "target_col": target_col,
            "train_size": str(train_size),
            "model": model_name,
            "metrics": metrics,
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "utility-evaluation-results",
        "Utility Evaluation results",
    )       

@flow(name="Anonymization Pipeline - Benchmark Dataset")
def anonymization_pipeline():

    file_extraction(input_file="raw_data.csv", output_file="ingested_raw_data.parquet", columns=(""),)

   
    direct_identifier_removal(
        input_file="ingested_raw_data.parquet",
        output_file="de_identified_data.parquet",
        identifier_columns=[
            "Patient_ID",
            "Full_Name",
            "Email",
            "Phone",
            "ZIP_Code",
            "Occupation",
        ],
    )


    anonymization(
        input_file="de_identified_data.parquet",
        output_file="age_anonymized.parquet",
        column="Age",
        strategy="numeric_interval",
        interval=10,
    )

    anonymization(
        input_file="age_anonymized.parquet",
        output_file="date_anonymized.parquet",
        column="Admission_Date",
        strategy="date",
        precision="year",
    )

    quasi_identifiers = [
        "Age",
        "Gender",
        "Marital_Status",
        "Admission_Date",
    ]

    k_anonymity(
        input_file="date_anonymized.parquet",
        output_file="k_anonymity_results.parquet",
        quasi_identifier_columns=quasi_identifiers,
        expected_k=5,
    )

    l_diversity(
        input_file="date_anonymized.parquet",
        output_file="l_diversity_results.parquet",
        quasi_identifier_columns=quasi_identifiers,
        sensitive_attribute="Diagnosis",
        expected_l=2,
    )

    t_closeness(
        input_file="date_anonymized.parquet",
        output_file="t_closeness_results.parquet",
        quasi_identifier_columns=quasi_identifiers,
        sensitive_attribute="Diagnosis",
        expected_t=0.2,
    )

    statistical_evaluation(
        real_data_file="de_identified_data.parquet",
        synthetic_data_file="date_anonymized.parquet",
        output_file="statistical_evaluation_results.parquet",
    )

    utility_evaluation(
        real_data_file="de_identified_data.parquet",
        synthetic_data_file="date_anonymized.parquet",
        feature_columns=["Age", "Gender", "Marital_Status", "Diagnosis", "Treatment"],
        target_col="Outcome",
        train_size=0.7,     
        model_name="SVC",
        metrics="accuracy, f1, roc_auc",
        output_file="utility_evaluation_results.parquet",
    )


if __name__ == "__main__":
    anonymization_pipeline()
