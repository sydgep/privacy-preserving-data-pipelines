# Reproducibility Guide

[← Main README](../README.md) · [Components Guide](components-guide.md)

This guide provides step-by-step instructions for reproducing the **Gaussian Copula**, **CTGAN**, and **CopulaGAN** experiments using Docker and Prefect. Each experiment includes **statistical fidelity**, **machine learning utility**, and **privacy risk evaluations**.

## 1. Prerequisites

Ensure the following tools are installed:

- **Python 3.11 or 3.12** (recommended).
- **Docker Engine or Docker Desktop**, installed and running.

Verify your installation:

```bash
python --version
docker --version
docker info
```

Use `python3` instead of `python` if required by your system. The `docker info` command must execute successfully.

An internet connection is required to install Python dependencies and download Docker images from GitHub Container Registry (GHCR).

## 2. Create a Virtual Environment

Create and activate a Python virtual environment.

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

## 3. Install Python Dependencies

Install the required Python libraries:

```bash
python -m pip install --upgrade pip
python -m pip install "prefect==3.6.20" docker pandas pyarrow
```

Prefect manages workflow orchestration, while the Docker Python SDK enables container execution. Pandas and PyArrow support reading and displaying Parquet outputs.

Other processing dependencies are packaged within the respective Docker images.

## 4. Start the Prefect Server

In the current terminal, start the local Prefect server:

```bash
prefect server start
```

Keep this terminal running throughout the experiment.

The Prefect dashboard is available at:

http://127.0.0.1:4200

## 5. Download the Dataset and Workflow

Download the following two files from the repository:

1. **[Artificial Dataset — data.csv](../Datasets/data.csv)**  
   Input dataset used for the experiments.

2. **[Prefect Workflow — artificial_data_pipeline.py](../Workflows/artificial_data_pipeline.py)**  
   Python script defining the complete generation and evaluation pipeline.

Save both files in a convenient location on your computer.

**Alternative: Clone the repository**

If you prefer to obtain the complete repository, run:

```bash
git clone https://github.com/sydgep/privacy-preserving-data-pipelines.git
cd privacy-preserving-data-pipelines
```

The dataset and workflow are available in the `Datasets/` and `Workflows/` directories, respectively.

## 6. Configure Dataset and Output Directories

Open `artificial_data_pipeline.py` and locate the following variables near the beginning of the script:

```python
DATASET_DIR = "/absolute/path/to/Experiments/Datasets"
HOST_SHARED_DIR = "/absolute/path/to/Experiments/outputs"
```

Replace these example paths with the actual absolute paths on your computer.

- **`DATASET_DIR`** specifies the directory containing `data.csv`, not the CSV file itself.
- **`HOST_SHARED_DIR`** specifies where intermediate datasets, generated synthetic data, and evaluation results will be stored. The workflow creates this directory if necessary.
- Both paths must be accessible to Docker. The input directory is mounted as read-only, while the output directory is mounted with read/write access.

**Windows users:** Use raw Python strings for Windows paths, for example:

```python
DATASET_DIR = r"C:\Users\username\Experiments\Datasets"
HOST_SHARED_DIR = r"C:\Users\username\Experiments\outputs"
```

Ensure Docker Desktop is configured to use Linux containers.

## 7. Execute the Workflow

Open a second terminal and activate the same Python virtual environment.

Configure the Prefect API connection:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Navigate to the directory containing the downloaded workflow and execute:

```bash
python artificial_data_pipeline.py
```

If you cloned the repository and are executing from its root directory, use:

```bash
python Workflows/artificial_data_pipeline.py
```

The workflow executes the data preparation, synthetic data generation, and evaluation tasks sequentially.

Docker containers are used to execute the individual processing components, while Prefect manages their orchestration and monitoring.

## 8. Monitor Execution and Verify Results

Open the Prefect dashboard at http://127.0.0.1:4200.

Under **Flow Runs**, select the **Artificial Data Pipeline** run to inspect task execution states and logs.

The **Artifacts** section provides previews of intermediate datasets and evaluation results.

After successful execution, the configured `HOST_SHARED_DIR` contains:

- Preprocessed training and test datasets.
- Three synthetic datasets generated using Gaussian Copula, CTGAN, and CopulaGAN.
- Statistical fidelity, machine learning utility, and privacy evaluation results for each generator.

The synthetic datasets are saved as:

```text
synthetic_data_copula.parquet
synthetic_data_ctgan.parquet
synthetic_data_copula_gan.parquet
```

Evaluation results are also saved in Parquet format.

## 9. Troubleshooting

| Problem | Suggested solution |
|---|---|
| Docker daemon unavailable | Ensure Docker is running and verify access using `docker info`. |
| Input dataset not found | Confirm that `DATASET_DIR` points to the directory containing `data.csv`. |
| Docker image download fails | Check internet connectivity and access to GitHub Container Registry. |
| Prefect dashboard is empty | Confirm that the server is running and `PREFECT_API_URL` points to `http://127.0.0.1:4200/api`. |
| Cannot write output files | Verify directory permissions and Docker file-sharing settings. |
| Missing Python dependency | Activate the virtual environment and repeat the dependency installation command. |
| Workflow script not found | Verify the current working directory and the path to `artificial_data_pipeline.py`. |

## 10. Reproducibility Notes

The experiments use a common training and test split to support comparisons across the three synthetic data generators.

For details about the individual components, their configuration parameters, and standalone execution, consult the [Components Guide](components-guide.md).
