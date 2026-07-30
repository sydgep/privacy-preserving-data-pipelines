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
HOST_SHARED_DIR = "/path/to/Datasets/outputs/brca_synth"
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
        "ingested-data",
        f"Raw dataset ingested from {input_file}",
    )

@task(name="Data Cleaning")
def drop_missing_values(input_file, output_file):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/drop_nan:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "cleaned-data",
        "Dataset after dropping rows with missing values",
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
    


@task(name="Synthetic Data Generation - VAE")
def vae_synthesizer(input_file, output_file):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/vae_synthesizer:v1",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "synthetic-data",
        "Generated synthetic dataset",
    )

@task(name="Synthetic Data Generation - COPULA")
def copula_synthesizer(input_file, output_file):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/copula_synthesizer:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "synthetic-data",
        "Generated synthetic dataset",
    )

@task(name="Synthetic Data Generation - CTGAN")
def ctgan_synthesizer(input_file, output_file):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/ctgan_synthesizer:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "synthetic-data",
        "Generated synthetic dataset",
    )

@task(name="Attribute Inference Attack")
def attribute_inference_attack(
    real_data_file,
    synthetic_data_file,
    output_file,
    quasi_identifier_columns,
    sensitive_attribute,
    model="random_forest",
):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/privacy_evaluation_aia:latest",
        environment={
            "container_input_real": f"{CONTAINER_OUTPUT_DIR}/{real_data_file}",
            "container_input_synth": f"{CONTAINER_OUTPUT_DIR}/{synthetic_data_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "quasi_identifier_columns": json.dumps(quasi_identifier_columns),
            "sensitive_attribute": sensitive_attribute,
            "model": model,
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "aia-results",
        "Attribute Inference Attack results",
    )


@task(name="Record Linkage Risk")
def record_linkage_risk(real_data_file, synthetic_data_file, output_file, quasi_identifier_columns):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/privacy_evaluation_rlr:latest",
        environment={
            "container_input_real": f"{CONTAINER_OUTPUT_DIR}/{real_data_file}",
            "container_input_synth": f"{CONTAINER_OUTPUT_DIR}/{synthetic_data_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "quasi_identifier_columns": json.dumps(quasi_identifier_columns),
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "rlr-results",
        "Record Linkage Risk results",
    )


@task(name="DCR Evaluation")
def dcr_evaluation(real_data_file, synthetic_data_file, output_file):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/privacy_evaluation_dcr:latest",
        environment={
            "container_input_real": f"{CONTAINER_OUTPUT_DIR}/{real_data_file}",
            "container_input_synth": f"{CONTAINER_OUTPUT_DIR}/{synthetic_data_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "dcr-results",
        "Distance to Closest Record results",
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

@flow(name="Synthesization Pipeline - BRCA Dataset")
def synthesization_pipeline():

    
    file_extraction(
        input_file="BRCA.csv",
        output_file="brca_raw.parquet",
        columns=(
            "Patient_ID,Age,Gender,Protein1,Protein2,Protein3,Protein4,"
            "Tumour_Stage,Histology,ER status,PR status,HER2 status,"
            "Surgery_type,Date_of_Surgery,Date_of_Last_Visit,Patient_Status"
        ),
    )

    drop_missing_values(
        input_file="brca_raw.parquet",
        output_file="brca_cleaned.parquet",
    )

    direct_identifier_removal(
        input_file="brca_cleaned.parquet",
        output_file="de_identified_data.parquet",
        identifier_columns=["Patient_ID"],
    )

    #vae_synthesizer(input_file="de_identified_data.parquet", output_file="synthetic_data.parquet")
    #copula_synthesizer(input_file="de_identified_data.parquet", output_file="synthetic_data.parquet")
    ctgan_synthesizer(input_file="de_identified_data.parquet", output_file="synthetic_data.parquet")

    quasi_identifiers = ["Age", "Gender", "Date_of_Surgery", "Date_of_Last_Visit"]

    
    attribute_inference_attack(
        real_data_file="de_identified_data.parquet",
        synthetic_data_file="synthetic_data.parquet",
        output_file="aia_results.parquet",
        quasi_identifier_columns=quasi_identifiers,
        sensitive_attribute="Patient_Status",
    )

    record_linkage_risk(
        real_data_file="de_identified_data.parquet",
        synthetic_data_file="synthetic_data.parquet",
        output_file="rlr_results.parquet",
        quasi_identifier_columns=quasi_identifiers,
    )

    dcr_evaluation(
        real_data_file="de_identified_data.parquet",
        synthetic_data_file="synthetic_data.parquet",
        output_file="dcr_results.parquet",
    )

    statistical_evaluation(
        real_data_file="de_identified_data.parquet",
        synthetic_data_file="synthetic_data.parquet",
        output_file="statistical_evaluation_results.parquet",
    )

    utility_evaluation(
        real_data_file="de_identified_data.parquet",
        synthetic_data_file="synthetic_data.parquet",
        output_file="utility_evaluation_results.parquet",
        feature_columns=["Age", "Gender", "Protein1", "Protein2", "Protein3", "Protein4", "Tumour_Stage", "Histology", "ER status", "PR status", "HER2 status", "Surgery_type"],
        target_col="Patient_Status",
        train_size=0.7,
        model_name="RandomForestClassifier",
        metrics="accuracy, f1, roc_auc")


if __name__ == "__main__":
    synthesization_pipeline()
