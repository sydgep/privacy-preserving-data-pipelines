# Workflows

This directory contains the six Prefect workflows used to execute and evaluate the privacy-preserving healthcare data pipelines described in the accompanying research work.

The workflows cover two privacy-preserving approaches—anonymization and synthetic data generation—applied to three datasets. Each workflow orchestrates independently executable Docker components. The Docker image references and component parameters are already defined within the corresponding Python workflow files.

This guide explains how to:

* prepare the local Python environment;
* install the required Python libraries;
* configure input and output directories;
* start the local Prefect server;
* connect the workflows to the Prefect API;
* execute and monitor a pipeline; and
* troubleshoot common execution problems.

---

## 1. Prerequisites

Before executing a workflow, ensure that the following software is available:

* Python 3.10 or later;
* Docker Engine or Docker Desktop;
* Git, when the repository is cloned from GitHub;
* sufficient disk space for input datasets, intermediate files, outputs, and Docker images.

The Docker daemon must be running because the workflow uses the Docker Python SDK to create and execute containers.

Verify the installations with:

```bash
python --version
docker --version
docker info
```

The `docker info` command should complete successfully. An error indicating that the Docker daemon cannot be reached usually means Docker is not running or the current user does not have permission to access it.

---

## 2. Clone the Repository

Clone the repository and enter the workflow directory:

```bash
git clone https://github.com/INSAFEDARE/privacy-preserving-framework.git
cd privacy-preserving-framework/Workflows
```


---

## 3. Create a Virtual Environment

A virtual environment isolates the Python packages required to run the workflows from the system-wide Python installation.

### Linux and macOS

From the repository root or the `workflows` directory, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Depending on the system configuration, `python` may be used instead of `python3`:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

After activation, the command prompt should normally include the environment name:

```text
(.venv)
```

Upgrade `pip` before installing the required libraries:

```bash
python -m pip install --upgrade pip
```

To leave the virtual environment later, run:

```bash
deactivate
```

---

## 4. Install the Required Python Libraries

This project does not use a `requirements.txt` file. Install the workflow dependencies directly:

```bash
python -m pip install "prefect==3.6.20" docker pandas
```

The workflow files contain the following imports:

```python
import os
import json
import docker
import pandas as pd

from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact
```

The `os` and `json` modules are included in the Python standard library and do not require separate installation.

The externally installed packages are:

| Package           | Purpose                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| `prefect==3.6.20` | Workflow orchestration, task execution, logging, state management, and artifacts |
| `docker`          | Communication with the local Docker daemon and execution of component containers |
| `pandas`          | Local data handling and preparation of tabular artifact content                  |

Verify the installations:

```bash
python -c "import prefect; print('Prefect:', prefect.__version__)"
python -c "import docker; print('Docker SDK:', docker.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
```

---

## 5. Configure the Dataset and Output Directories

Before running a workflow, open the selected Python file and update its directory configuration.

A workflow typically contains variables similar to:

```python
DATASET_DIR = "/path/to/Datasets"
HOST_SHARED_DIR = "/path/to/Datasets/outputs/brca_anonym"
```

These paths are host-machine directories. They are mounted into the Docker containers when individual pipeline components execute.

### `DATASET_DIR`

`DATASET_DIR` identifies the directory containing the input dataset.

For example:

```python
DATASET_DIR = "/home/user/privacy-framework/datasets"
```

The corresponding input file might be:

```text
/home/user/privacy-framework/datasets/BRCA.csv
```

### `HOST_SHARED_DIR`

`HOST_SHARED_DIR` identifies the directory where the workflow stores the intermediate outputs;


For example:

```python
HOST_SHARED_DIR = "/home/user/privacy-framework/outputs/brca_anonym"
```

Use a separate output directory for every workflow to prevent files from different experiments from overwriting one another.

---

## 6. Verify Docker Access from Python

The workflow communicates with Docker through the Docker Python SDK.

Before starting Prefect, test this connection:

```bash
python -c "import docker; client = docker.from_env(); print(client.ping())"
```

A successful connection should print:

```text
True
```

You may also list the locally available images:

```bash
python -c "import docker; client = docker.from_env(); print([image.tags for image in client.images.list()])"
```

The workflow code already specifies the Docker image associated with each component. No image-building step is required to execute these workflows.

When an image is stored in a private container registry, authenticate with that registry before running the workflow.

---

## 7. Start the Local Prefect Server

Open a terminal, activate the virtual environment, and start the Prefect server:

```bash
source .venv/bin/activate
prefect server start
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
prefect server start
```

Keep this terminal open while the workflows are running.

By default, the Prefect user interface is available at:

```text
http://127.0.0.1:4200
```

The API endpoint used by the workflow client is:

```text
http://127.0.0.1:4200/api
```

Stop the server when required by pressing:

```text
Ctrl+C
```

---

## 11. Configure the Prefect API URL

Open a second terminal and activate the same virtual environment.

### Linux and macOS

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Configure Prefect to use the local server:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Verify the configuration:

```bash
prefect config view
```

The output should contain:

```text
PREFECT_API_URL='http://127.0.0.1:4200/api'
```

You can also inspect only the API URL:

```bash
prefect config view --show-sources
```

This configuration is generally stored in the active Prefect profile and does not need to be repeated for every pipeline execution.

---

## 8. Test the Prefect Connection

Confirm that the server is running by opening:

```text
http://127.0.0.1:4200
```

in a web browser.

You can also verify the configured API URL from Python:

```bash
python -c "from prefect.settings import get_current_settings; print(get_current_settings().api.url)"
```

The result should point to:

```text
http://127.0.0.1:4200/api
```

---

## 9. Execute a Workflow

The six workflows may be executed sequentially:

```bash
python artificial_anonymization_pipeline.py
python artificial_synthesization_pipeline.py
python brca_anonymization_pipeline.py
python brca_synthesization_pipeline.py
python mimic_anonymization_pipeline.py
python mimic_synthesization_pipeline.py
```

Before executing all workflows, confirm that:

* every workflow has a distinct `HOST_SHARED_DIR`;
* the required datasets are available;
* MIMIC-IV access requirements have been satisfied;
* sufficient disk space is available; and
* the Prefect server remains active.

Sequential execution is recommended for the first complete reproduction because it simplifies log inspection and resource management.

---

## 10. Monitor Workflow Execution

After starting a workflow, monitor it in two places.

### Terminal

The terminal displays:

* flow-run creation;
* task-run status;
* Docker container execution;
* component logs;
* output-file locations;
* warnings;
* retries; and
* exceptions.

### Prefect User Interface

Open:

```text
http://127.0.0.1:4200
```

The Prefect interface provides:

* flow-run status;
* task dependencies;
* task execution duration;
* execution logs;
* failed-task information;
* retries;
* generated artifacts; and
* historical workflow runs.

A successful workflow should finish with a `Completed` state.

---

## 11. Workflow Outputs

Workflow results are written to the configured `HOST_SHARED_DIR`.

For example:

```python
HOST_SHARED_DIR = "/home/user/privacy-preserving-framework/outputs/brca_anonymization"
```

Prefect table artifacts may also be displayed in the Prefect user interface. These artifacts summarize selected execution outputs, but the files stored under `HOST_SHARED_DIR` remain the primary pipeline outputs.

---

## 12. Data Privacy Considerations

The workflows run Docker containers on the local machine and access datasets through configured host-directory mounts.

Users remain responsible for:

* obtaining permission to access restricted datasets;
* complying with applicable data-use agreements;
* limiting access to the Prefect interface;
* protecting host input and output directories;
* avoiding the inclusion of sensitive record-level information in logs or artifacts; and
* securely deleting intermediate outputs when they are no longer required.

Do not upload restricted datasets, MIMIC-IV-derived datasets, sensitive workflow outputs, or credentials to the public repository.

The local Prefect interface should not be exposed to an untrusted network without appropriate authentication and access controls.

---


