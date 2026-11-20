# Workflow Orchestration Guide

[← Main README](../README.md) · [Components](components-guide.md) · [Reproducibility](reproducibility-guide.md)

This guide explains **how to connect reusable Docker components** into a custom workflow. For installation and running the published experiments, see the [Reproducibility Guide](reproducibility-guide.md). For Docker commands and configuration options for each component, see the [Components Guide](components-guide.md).

## 1. The central idea: one component's output is another's input

Each component runs in its own container. Containers do not need to communicate directly: they exchange files through a **shared host directory**.

```text
Input CSV
   │
   ▼
[File extraction] ── writes ingested_data.parquet ──┐
                                                    │
[De-identification] ◄── reads ingested_data.parquet ┘
   │
   └── writes de_identified_data.parquet
                │
                ▼
          [Data cleaning] → cleaned_data.parquet → ...
```

For every connection, make sure (1) the producer has completed, (2) the next component reads the exact file it wrote, and (3) both containers mount the same host output directory.

## 2. Choose how to coordinate the containers

**Option A — Run them manually:** Execute the `docker run` commands one after another. This is suitable for testing a component or a short chain. The [Components Guide](components-guide.md) provides ready-to-adapt commands. You are responsible for execution order and handling errors.

**Option B — Use Prefect:** Wrap the same Docker calls in Prefect tasks and call them in the required order. Prefect records task states, logs, and flow runs, while Docker still executes the actual processing. The reference implementation uses this approach.

## 3. Understand the Prefect pattern

The reference script, [`Workflows/artificial_data_pipeline.py`](../Workflows/artificial_data_pipeline.py), has three parts:

- **A shared Docker helper:** `run_container(image, environment, logger)` starts a container, waits for completion, captures logs, and raises an error if the container fails.
- **Component tasks:** Each `@task` specifies an image and the environment variables that its container needs.
- **The flow:** A `@flow` function calls the tasks in the intended order.

A simplified example of the *pattern* (illustrative, not a replacement for the reference script):

```python
from prefect import flow, task

@task
def ingest():
    # Call the file-extraction Docker image and write ingested_data.parquet
    ...

@task
def deidentify():
    # Read ingested_data.parquet and write de_identified_data.parquet
    ...

@flow
def my_pipeline():
    ingest()
    deidentify()  # Runs after ingestion completes
```

The `...` placeholders represent Docker calls; copy the actual task implementations and Docker configuration from the [reference script](../Workflows/artificial_data_pipeline.py) when building a runnable workflow. Because the reference flow calls its tasks synchronously, its three experiments run sequentially.

## 4. Build your own pipeline

Start from the result you want, then work backwards to identify the required input files.

1. **Select components.** Use the [Components Guide](components-guide.md) to identify images and required environment variables.
2. **Connect file paths.** Set each consumer's input to the producer's output, using the same shared Docker volume.
3. **Order tasks.** In a manual chain, run commands sequentially. In Prefect, call dependent tasks in order and wait for completion.
4. **Keep outputs distinct.** Give separate generators and evaluations different output filenames to prevent overwrites.
5. **Check failures.** Stop downstream execution if a container fails or an expected output is missing.

For example, you can replace CTGAN with another compatible generator after train/test splitting. The evaluation tasks need the resulting synthetic dataset and, where applicable, the same real training and testing datasets. The evaluation components do not need to know which generator produced the file.

## 5. Adapt the reference workflow

The easiest starting point is to copy [`artificial_data_pipeline.py`](../Workflows/artificial_data_pipeline.py) and edit the task calls inside its `@flow` function. To add a component, create a Prefect task following an existing task's structure: pass the image name, map its environment variables, and call the shared Docker helper.

Keep the Docker image interface and the file naming consistent. For experimental reproducibility, record the workflow commit and the container image digests; `:latest` alone does not identify an immutable image version.
