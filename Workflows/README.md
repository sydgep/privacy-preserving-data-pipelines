# Reproducibility Experiments

This guide provides step-by-step instructions for reproducing the synthetic data generation and evaluation experiments using Docker and Prefect.

The workflow executes three experiments sequentially, using **Gaussian Copula, CTGAN, and CopulaGAN** to generate synthetic datasets. Each generated dataset is subsequently evaluated in terms of statistical fidelity, machine learning utility, and privacy-related risks.

All processing components are executed as independent Docker containers, while Prefect manages workflow orchestration, execution monitoring, and visualization.

## 1. Prerequisites

Before executing the pipeline, ensure that the following software is installed:

- **Python 3.10 or later** (Python 3.11 or 3.12 recommended).
- **Docker Engine or Docker Desktop**.
- **Git** (recommended for downloading the repository).

The Docker daemon must be running because the workflow uses the Docker Python SDK to create and execute containers.

Verify the installations:

```bash
python --version
docker --version
docker info
```

On some Linux and macOS systems, use `python3` instead of `python`.

The `docker info` command should complete successfully. If it reports that the Docker daemon cannot be reached, ensure that Docker is running and that your user account has permission to access it.

**Note:** An internet connection is required to download Python libraries and pull the Docker images from GitHub Container Registry (GHCR). The pipeline automatically checks for the latest Docker images before executing each component.

## 2. Download the Dataset and Pipeline

Clone the GitHub repository:

```bash
git clone https://github.com/<OWNER>/<REPOSITORY>.git
cd <REPOSITORY>
```

Replace `<OWNER>` and `<REPOSITORY>` with the actual GitHub repository details.

Alternatively, download the repository as a ZIP file using **Code → Download ZIP** on GitHub, then extract it.

The repository includes:

- `artificial_data_pipeline.py` — the executable Prefect workflow.
- `data.csv` — the input dataset used in the experiments.

No additional dataset download is required.

## 3. Create a Virtual Environment

Creating a Python virtual environment is recommended to isolate the workflow dependencies from other Python projects.

### Linux and macOS

From the repository directory, execute:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

Once activated, the terminal should display `(venv)` before the command prompt.

## 4. Install the Required Python Libraries

With the virtual environment activated, install the required libraries:

```bash
python -m pip install "prefect=3.6.20" docker pandas pyarrow
```

These libraries provide the functionality required by the orchestration script:

| Library | Purpose |
|---|---|
| `prefect` | Workflow orchestration, monitoring, and artifact management |
| `docker` | Programmatic execution of Docker containers |
| `pandas` | Reading and processing Parquet outputs for Prefect artifacts |
| `pyarrow` | Parquet file support for pandas |

The remaining processing dependencies are packaged within their respective Docker images and do not need to be installed in the local Python environment.

## 5. Configure the Dataset and Output Directories

Before running the workflow, open `artificial_data_pipeline.py` using any text editor or IDE.

Locate **lines 11 and 12**:

```python
DATASET_DIR = "/homelocal/path/to/Datasets/Artificial"
HOST_SHARED_DIR = "/homelocal/path/to/Datasets/outputs"
```
Update these variables according to your local directory structure.

- `DATASET_DIR`: Absolute path to the directory containing `data.csv`.
- `HOST_SHARED_DIR`: Absolute path to the directory where intermediate datasets, synthetic datasets, and evaluation results will be stored.

## 6. Start the Local Prefect Server

Open a terminal in the repository directory and activate the virtual environment.

Start the local Prefect server:

```bash
prefect server start
```

By default, the Prefect user interface is accessible at:

http://127.0.0.1:4200

Keep this terminal running throughout the experiments.

## 7. Execute the Workflow

Open a **second terminal** in the repository directory.

Activate the same virtual environment.

### Linux and macOS

```bash
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

Configure the Prefect API endpoint:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Then execute the pipeline:

```bash
python artificial_data_pipeline.py
```

The workflow executes the following stages:

1. Data ingestion from `data.csv`.
2. Removal of specified direct identifiers.
3. Data cleaning.
4. Training and testing dataset splitting.
5. Synthetic data generation using Gaussian Copula, followed by statistical, utility, and privacy evaluation.
6. Synthetic data generation using CTGAN, followed by the same evaluations.
7. Synthetic data generation using CopulaGAN, followed by the same evaluations.

The three experiments are executed sequentially.

**Note:** The workflow automatically pulls the specified Docker images from GHCR before executing each component. The first execution may take longer because the required images must be downloaded.

## 8. Monitor Workflow Execution

Open the Prefect dashboard in a web browser:

http://127.0.0.1:4200

From the dashboard:

1. Navigate to **Flow Runs**.
2. Select the **Artificial Data Pipeline** flow run.
3. Inspect the execution status of individual tasks.
4. Review task logs to monitor progress or investigate failures.
5. Open **Artifacts** to inspect previews of generated datasets and evaluation results.

The dashboard provides a graphical overview of workflow execution and allows users to monitor the different processing stages.

## 9. Workflow Outputs

All generated files are stored in the directory specified by `HOST_SHARED_DIR`.

### Data Preparation

| Output | Description |
|---|---|
| `ingested_data.parquet` | Dataset after ingestion |
| `de_identified_data.parquet` | Dataset after direct identifier removal |
| `cleaned_data.parquet` | Dataset after missing-value cleaning |
| `train_data.parquet` | Real training dataset |
| `test_data.parquet` | Held-out real testing dataset |

### Synthetic Data Generation

| Output | Description |
|---|---|
| `synthetic_data_copula.parquet` | Synthetic dataset generated using Gaussian Copula |
| `synthetic_data_ctgan.parquet` | Synthetic dataset generated using CTGAN |
| `synthetic_data_copula_gan.parquet` | Synthetic dataset generated using CopulaGAN |

### Evaluation Results

Each synthetic dataset is evaluated using three complementary evaluation components.

| Evaluation | Gaussian Copula | CTGAN | CopulaGAN |
|---|---|---|---|
| Statistical fidelity | `statistical_evaluation_results_copula.parquet` | `statistical_evaluation_results_ctgan.parquet` | `statistical_evaluation_results_copula_gan.parquet` |
| Machine learning utility | `utility_evaluation_results_copula.parquet` | `utility_evaluation_results_ctgan.parquet` | `utility_evaluation_results_copula_gan.parquet` |
| Privacy evaluation | `privacy_evaluation_results_copula.parquet` | `privacy_evaluation_results_ctgan.parquet` | `privacy_evaluation_results_copula_gan.parquet` |

The evaluation components assess:

- **Statistical fidelity:** Distribution and correlation similarity between real and synthetic datasets.
- **Machine learning utility:** Predictive performance of models trained on real versus synthetic data and evaluated on held-out real data.
- **Privacy:** Singling-out, linkability, and inference attacks, with estimated privacy risks and corresponding attack success rates.

## 10. Troubleshooting

**Docker daemon not accessible**

Ensure that Docker is running and that the current user has permission to access the Docker daemon.

**Input dataset not found**

Verify that `DATASET_DIR` points to the directory containing `data.csv`, rather than to the CSV file itself.

**Docker image download fails**

Verify internet connectivity and access to the required images on GHCR.

**Prefect dashboard does not display the workflow**

Confirm that the Prefect server is running and that `PREFECT_API_URL` points to `http://127.0.0.1:4200/api`.

**Permission denied when accessing output files**

Ensure that Docker can read the input directory and write to the output directory.

**Missing Python module**

Confirm that the virtual environment is activated and repeat the installation command in Section 4.

---

## Reproducibility Notes

The experiments use a common real training dataset and held-out real testing dataset to support comparisons across the three synthetic data generation approaches.

Although the pipeline specifies a random seed for the train/test split, synthetic data generation methods may exhibit variability between executions, depending on their internal random initialization and computational environment.

For exact reproduction of published results, record the versions or image digests of the Docker images used during the experiments.
