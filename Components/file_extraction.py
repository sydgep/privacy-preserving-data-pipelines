import os
import json
import pandas as pd


def parse_columns(columns_raw):
    """
    Parses column names from JSON or comma-separated format.
    Returns None if input is missing or empty, so pandas reads all columns.
    """
    if not columns_raw:
        return None

    try:
        parsed = json.loads(columns_raw)
        if isinstance(parsed, list) and parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: comma-separated string
    split = [col.strip() for col in columns_raw.split(",") if col.strip()]
    return split if split else None


def detect_compression(filepath):
    """
    Detects compression method based on file extension.
    """
    if filepath.endswith(".gz"):
        return "gzip"
    elif filepath.endswith(".zip"):
        return "zip"
    return None


def file_extraction(container_input, container_output, columns):
    """
    Ingests data from various formats and writes to a Parquet file.
    Supported formats: CSV, JSON, Parquet, TXT, Excel (XLS/XLSX).
    Supports gzip and zip compression.
    """
    if not container_input or not container_output:
        raise ValueError("Both container_input and container_output must be specified.")

    compression = detect_compression(container_input)

    if container_input.endswith((".csv", ".csv.gz", ".csv.zip")):
        df = pd.read_csv(container_input, usecols=columns, compression=compression, low_memory=False)

    elif container_input.endswith((".txt", ".txt.gz", ".txt.zip")):
        df = pd.read_csv(container_input, delimiter="\t", usecols=columns, compression=compression, low_memory=False)

    elif container_input.endswith(".json") or container_input.endswith(".json.gz"):
        df = pd.read_json(container_input, compression=compression)
        if columns:
            df = df[columns]

    elif container_input.endswith(".parquet"):
        df = pd.read_parquet(container_input, columns=columns)

    elif container_input.endswith((".xls", ".xlsx")):
        df = pd.read_excel(container_input, usecols=columns)

    else:
        raise ValueError(f"Unsupported file type: {container_input}")

    df.to_parquet(container_output, index=False)
    print(f"✅ File successfully ingested and saved → {container_output}")


if __name__ == "__main__":
    container_input = os.environ.get("container_input")
    container_output = os.environ.get("container_output")
    columns_raw = os.environ.get("columns", "")

    columns = parse_columns(columns_raw)
    file_extraction(container_input, container_output, columns)

