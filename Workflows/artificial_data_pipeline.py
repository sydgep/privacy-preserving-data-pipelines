import os
import json
import docker
import pandas as pd
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
DATASET_DIR = "/homelocal/ms271520/INSAFEDARE/Datasets/Artificial"
HOST_SHARED_DIR = "/homelocal/ms271520/INSAFEDARE/Datasets/outputs/artificial_data_pipeline"
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
        f"Dataset ingested from {input_file}",
    )

@task(name="De-identification")
def de_identification(input_file, output_file, identifier_columns):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/de_identification:latest",
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
        "De-identified dataset after removal of direct identifiers",
    )

@task(name="Data cleaning") 
def data_cleaning(input_file, output_file):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/missing_value_removal:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "cleaned-data",
        "Dataset after cleaning of missing representations such as NaN, None, NA, N/A, NULL, MISSING, whitespace and empty strings",
    )
    
@task(name="Train-Test Split")
def train_test_split(input_file, output_file_train, output_file_test, train_size, stratify, seed):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/train_test_split:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output_train": f"{CONTAINER_OUTPUT_DIR}/{output_file_train}",
            "container_output_test": f"{CONTAINER_OUTPUT_DIR}/{output_file_test}",
            "train_size": train_size,
            "stratify": stratify,
            "seed": seed,
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file_train,
        "training-data",
        "Training dataset after split",
    )

    create_parquet_artifact(
        output_file_test,
        "testing-data",
        "Testing dataset after split",
    )
        

@task(name="Synthetic Data Generation - Gaussian Copula")
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
        "synthetic-data-copula",
        "Generated synthetic dataset using gaussian copula synthesizer",
    )

@task(name="Synthetic Data Generation - CTGAN")
def ctgan_synthesizer(input_file, output_file, epochs, batch_size):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/ctgan_synthesizer:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "epochs": epochs,
            "batch_size": batch_size,
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "synthetic-data-ctgan",
        "Generated synthetic dataset using CTGAN synthesizer",
    )

@task(name="Synthetic Data Generation - CopulaGAN")
def copula_gan_synthesizer(input_file, output_file, epochs, batch_size):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/copula_gan_synthesizer:latest",
        environment={
            "container_input": f"{CONTAINER_OUTPUT_DIR}/{input_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "epochs": epochs,
            "batch_size": batch_size,
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "synthetic-data-copula-gan",
        "Generated synthetic dataset using CopulaGAN synthesizer",
    )

@task(name="Statistical Evaluation")
def statistical_evaluation(real_data_file, synthetic_data_file, output_file):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/statistical_evaluation:latest",
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
def utility_evaluation(
    real_train_file,
    real_test_file,
    synthetic_data_file,
    output_file,
    feature_columns,
    target_col,
    task_type,
    model_name,
    metrics,
):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/ml_utility_evaluation:latest",
        environment={
            "container_input_real_train": f"{CONTAINER_OUTPUT_DIR}/{real_train_file}",
            "container_input_real_test": f"{CONTAINER_OUTPUT_DIR}/{real_test_file}",
            "container_input_synth": f"{CONTAINER_OUTPUT_DIR}/{synthetic_data_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "feature_columns": feature_columns,
            "target_col": target_col,
            "task_type": task_type,
            "model_name": model_name,
            "metrics": metrics,
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "utility-evaluation-results",
        "Utility Evaluation results",
    )

@task(name="Privacy Evaluation")
def privacy_evaluation(
    real_train_file,
    real_test_file,
    synthetic_data_file,
    output_file,
    quasi_identifiers,
    sensitive_attributes,
    linkability_aux_cols_1,
    linkability_aux_cols_2,
):
    logger = get_run_logger()

    run_container(
        image="ghcr.io/sydgep/docker-images/privacy_attack_evaluation:latest",
        environment={
            "container_input_real_train": f"{CONTAINER_OUTPUT_DIR}/{real_train_file}",
            "container_input_real_test": f"{CONTAINER_OUTPUT_DIR}/{real_test_file}",
            "container_input_synth": f"{CONTAINER_OUTPUT_DIR}/{synthetic_data_file}",
            "container_output": f"{CONTAINER_OUTPUT_DIR}/{output_file}",
            "quasi_identifiers": quasi_identifiers,
            "sensitive_attributes": sensitive_attributes,
            "linkability_aux_cols_1": linkability_aux_cols_1,
            "linkability_aux_cols_2": linkability_aux_cols_2,
        },
        logger=logger,
    )

    create_parquet_artifact(
        output_file,
        "privacy-evaluation-results",
        "Privacy Evaluation results",
    )

@flow(name="Artificial Data Pipeline")
def synthesization_pipeline():
    
    file_extraction(input_file="data.csv", output_file="ingested_data.parquet", columns=(""))

    de_identification(
        input_file="ingested_data.parquet",
        output_file="de_identified_data.parquet",
        identifier_columns=[
            "Patient_ID",
            "Full_Name",
            "Email",
            "Phone",
            "ZIP_Code",
        ],
    )

    data_cleaning(
        input_file="de_identified_data.parquet",
        output_file="cleaned_data.parquet",
    )

    train_test_split(
        input_file="cleaned_data.parquet",
        output_file_train="train_data.parquet",
        output_file_test="test_data.parquet",
        train_size=0.7,
        stratify="Outcome",
        seed=42,
    )

    # First Experiment: Gaussian Copula Synthesizer (A statistical model-based approach)

    copula_synthesizer(input_file="train_data.parquet", output_file="synthetic_data_copula.parquet")

    statistical_evaluation(
        real_data_file="train_data.parquet",
        synthetic_data_file="synthetic_data_copula.parquet",
        output_file="statistical_evaluation_results_copula.parquet",
    )    


    utility_evaluation(
        real_train_file="train_data.parquet",
        real_test_file="test_data.parquet",
        synthetic_data_file="synthetic_data_copula.parquet",
        output_file="utility_evaluation_results_copula.parquet",
        feature_columns="Age, Gender, Marital_Status, Diagnosis, Treatment",
        target_col="Outcome", 
        task_type="classification",
        model_name="GradientBoostingClassifier",
        metrics="accuracy, f1, roc_auc",
    )

    privacy_evaluation(
        real_train_file="train_data.parquet",
        real_test_file="test_data.parquet",
        synthetic_data_file="synthetic_data_copula.parquet",
        output_file="privacy_evaluation_results_copula.parquet",
        quasi_identifiers="Age, Gender, Marital_Status, Occupation, Admission_Date",
        sensitive_attributes="Diagnosis, Treatment, Outcome",
        linkability_aux_cols_1="Age, Gender, Occupation",
        linkability_aux_cols_2="Admission_Date, Diagnosis, Treatment",
  )    

    # Second Experiment: CTGAN Synthesizer (A deep learning-based approach)
    
    ctgan_synthesizer(input_file="train_data.parquet", output_file="synthetic_data_ctgan.parquet", epochs=300, batch_size=100)

    statistical_evaluation(
        real_data_file="train_data.parquet",
        synthetic_data_file="synthetic_data_ctgan.parquet",
        output_file="statistical_evaluation_results_ctgan.parquet",
    )

    utility_evaluation(
        real_train_file="train_data.parquet",
        real_test_file="test_data.parquet",
        synthetic_data_file="synthetic_data_ctgan.parquet",
        output_file="utility_evaluation_results_ctgan.parquet",
        feature_columns="Age, Gender, Marital_Status, Diagnosis, Treatment",
        target_col="Outcome",
        task_type="classification",
        model_name="GradientBoostingClassifier",
        metrics="accuracy, f1, roc_auc",
    )

    privacy_evaluation(
        real_train_file="train_data.parquet",
        real_test_file="test_data.parquet",
        synthetic_data_file="synthetic_data_ctgan.parquet",
        output_file="privacy_evaluation_results_ctgan.parquet",
        quasi_identifiers="Age, Gender, Marital_Status, Occupation, Admission_Date",
        sensitive_attributes="Diagnosis, Treatment, Outcome",
        linkability_aux_cols_1="Age, Gender, Occupation",
        linkability_aux_cols_2="Admission_Date, Diagnosis, Treatment",
    )

    # Third Experiment: CopulaGAN Synthesizer (A hybrid approach combining statistical and deep learning techniques)

    copula_gan_synthesizer(input_file="train_data.parquet", output_file="synthetic_data_copula_gan.parquet", epochs=300, batch_size=100)

    statistical_evaluation(
        real_data_file="train_data.parquet",
        synthetic_data_file="synthetic_data_copula_gan.parquet",
        output_file="statistical_evaluation_results_copula_gan.parquet",
    )

    utility_evaluation(
        real_train_file="train_data.parquet",
        real_test_file="test_data.parquet",
        synthetic_data_file="synthetic_data_copula_gan.parquet",
        output_file="utility_evaluation_results_copula_gan.parquet",
        feature_columns="Age, Gender, Marital_Status, Diagnosis, Treatment",
        target_col="Outcome",
        task_type="classification",
        model_name="GradientBoostingClassifier",
        metrics="accuracy, f1, roc_auc",
    )

    privacy_evaluation(
        real_train_file="train_data.parquet",
        real_test_file="test_data.parquet",
        synthetic_data_file="synthetic_data_copula_gan.parquet",
        output_file="privacy_evaluation_results_copula_gan.parquet",
        quasi_identifiers="Age, Gender, Marital_Status, Occupation, Admission_Date",
        sensitive_attributes="Diagnosis, Treatment, Outcome",
        linkability_aux_cols_1="Age, Gender, Occupation",
        linkability_aux_cols_2="Admission_Date, Diagnosis, Treatment",
    )


if __name__ == "__main__":
    synthesization_pipeline()
