# Reproducibility Guide

[← Main README](../README.md) · [Components](components-guide.md) · [Orchestration](workflow-orchestration.md)

This guide reproduces the sequential **Gaussian Copula**, **CTGAN**, and **CopulaGAN** experiments on the supplied artificial dataset. Each generator is followed by statistical fidelity, machine learning utility, and privacy evaluation.

## 1. Prerequisites

Install **Python 3.11 or 3.12** (recommended), **Docker Engine or Docker Desktop**, and **Git** (optional if downloading ZIP). Docker must be running and accessible to the account executing Python.

```bash
python --version
docker --version
docker info
```

Use `python3` in place of `python` if required. `docker info` must succeed. Internet access is required for Python packages and GHCR container images.

## 2. Download the repository

```bash
git clone https://github.com/sydgep/privacy-preserving-data-pipelines.git
cd privacy-preserving-data-pipelines
```

For a private repository, your GitHub account must have access. Alternatively, download the repository ZIP and extract it. Confirm that `Datasets/data.csv` and `Workflows/artificial_data_pipeline.py` are present.

## 3. Create a virtual environment

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

## 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install "prefect==3.6.20" docker pandas pyarrow
```

`prefect` handles orchestration and artifacts; `docker` provides the Python Docker SDK; `pandas` and `pyarrow` support Parquet artifact previews. Other processing libraries are provided by the respective Docker images. If this Prefect version is unavailable for your Python installation, use a compatible Python 3.11/3.12 environment rather than silently changing the documented experiment environment.

## 5. Configure the dataset and output directories

Open `Workflows/artificial_data_pipeline.py` and locate `DATASET_DIR` and `HOST_SHARED_DIR` near the beginning:

```python
DATASET_DIR = "/absolute/path/to/privacy-preserving-data-pipelines/Datasets"
HOST_SHARED_DIR = "/absolute/path/to/privacy-preserving-data-pipelines/outputs"
```

- `DATASET_DIR` is the **directory** containing `data.csv`, not the CSV file itself.
- `HOST_SHARED_DIR` is the destination for intermediate data, synthetic datasets, and evaluation results. The script creates this directory if necessary.
- Use **absolute host paths** and ensure Docker has access to both locations. The input mount is read-only; the output mount is read/write.
- On Windows, use raw Python strings for paths, for example `r"C:\Users\username\privacy-preserving-data-pipelines\Datasets"`, and run Docker Desktop in Linux-container mode.

## 6. Start Prefect (terminal 1)

Activate the environment and run:

```bash
prefect server start
```

Leave this terminal open. The dashboard is usually available at <http://127.0.0.1:4200>.

## 7. Run the workflow (terminal 2)

Open another terminal in the **repository root** and activate the same virtual environment. Point the Prefect client at the local server:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
python Workflows/artificial_data_pipeline.py
```

The pipeline executes, in order: file ingestion → direct-identifier removal → missing-value cleaning → stratified train/test split → Gaussian Copula and its three evaluations → CTGAN and its three evaluations → CopulaGAN and its three evaluations.

The first run may be slower while images are downloaded. **Image update behavior depends on the committed `run_container()` implementation:** if it calls `docker_client.images.pull(image)` before execution, it checks GHCR on every task; otherwise a locally cached image may be reused. Inspect the committed file before relying on automatic updates.

## 8. Monitor execution

Open <http://127.0.0.1:4200>, select **Flow Runs**, and open **Artificial Data Pipeline**. Inspect task states and logs; use **Artifacts** for previews of Parquet outputs and evaluation tables.

## 9. Check the outputs

Successful execution creates intermediate datasets, three synthetic datasets (`synthetic_data_copula.parquet`, `synthetic_data_ctgan.parquet`, and `synthetic_data_copula_gan.parquet`), and statistical, utility, and privacy evaluation results for each generator in `HOST_SHARED_DIR`. The Prefect **Artifacts** page also provides previews. Check the output directory and task logs if an expected result is missing.

## 10. Troubleshooting

| Problem | Check |
|---|---|
| Docker daemon unavailable | Start Docker; check `docker info` and user permissions |
| Input file not found | Ensure `DATASET_DIR` points to the folder containing `data.csv` |
| Image pull fails | Check network connectivity and GHCR image access |
| Prefect dashboard empty | Verify server is running and `PREFECT_API_URL` points to `/api` |
| Cannot write output | Check output directory permissions and Docker file-sharing settings |
| Missing Python package | Activate the environment and repeat the installation command |
| Script not found | Run `python Workflows/artificial_data_pipeline.py` from repository root |

## Reproducibility considerations

The split uses a configured seed, but the synthetic generators may still be stochastic. For paper-level reproducibility, record container image **digests**, the dataset revision, pipeline commit, and package versions. Pulling mutable `:latest` tags is useful during development but cannot guarantee identical future runs.
