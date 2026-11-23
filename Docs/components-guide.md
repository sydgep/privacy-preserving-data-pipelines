# Reusable Components Guide

[← Main README](../README.md) · [Reproducibility](reproducibility-guide.md) ·

Each component is distributed as a Docker image and configured through environment variables. You can run a component on its own; no Prefect installation is needed for the Docker commands below.

The following are the **ten components used by the reference experiment**. For a complete step-by-step experiment, use the [Reproducibility Guide](reproducibility-guide.md).

## 1. File Extraction

**Functionality:** Reads a source file and produces a Parquet dataset. The `columns` setting can select a subset; an empty value requests all columns.

**Docker image:** `ghcr.io/sydgep/docker-images/file_extraction:latest`

**Configuration:** `container_input` (source path); `container_output` (Parquet destination); `columns` (column selection).

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/Datasets:/app/datasets:ro" -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input=/app/datasets/data.csv \
  -e container_output=/app/outputs/ingested_data.parquet \
  -e columns="" \
  ghcr.io/sydgep/docker-images/file_extraction:latest
```

**Expected primary output:** `outputs/ingested_data.parquet`.

## 2. De-identification

**Functionality:** Removes the specified direct-identifier columns. This is column removal, not a guarantee of anonymization.

**Docker image:** `ghcr.io/sydgep/docker-images/de_identification:latest`

**Configuration:** `container_input`; `container_output`; `identifier_columns` (JSON list).

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input=/app/outputs/ingested_data.parquet \
  -e container_output=/app/outputs/de_identified_data.parquet \
  -e 'identifier_columns=["Patient_ID","Full_Name","Email","Phone","ZIP_Code"]' \
  ghcr.io/sydgep/docker-images/de_identification:latest
```

**Expected primary output:** `outputs/de_identified_data.parquet`.

## 3. Missing-value Removal

**Functionality:** Cleans missing-value representations in the input dataset. The reference workflow uses it before splitting the data.

**Docker image:** `ghcr.io/sydgep/docker-images/missing_value_removal:latest`

**Configuration:** `container_input`; `container_output`.

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input=/app/outputs/de_identified_data.parquet \
  -e container_output=/app/outputs/cleaned_data.parquet \
  ghcr.io/sydgep/docker-images/missing_value_removal:latest
```

**Expected primary output:** `outputs/cleaned_data.parquet`.

## 4. Train/Test Split

**Functionality:** Partitions the cleaned data into real training and held-out testing sets. The workflow configures a 70/30 split and stratification on `Outcome`.

**Docker image:** `ghcr.io/sydgep/docker-images/train_test_split:latest`

**Configuration:** `container_input`; `container_output_train`; `container_output_test`; `train_size`; `stratify`; `stratify_method`; `group_column`; `seed`.

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input=/app/outputs/cleaned_data.parquet \
  -e container_output_train=/app/outputs/train_data.parquet \
  -e container_output_test=/app/outputs/test_data.parquet \
  -e train_size=0.7 \
  -e stratify=Outcome \
  -e stratify_method=max \
  -e group_column="" \
  -e seed=42 \
  ghcr.io/sydgep/docker-images/train_test_split:latest
```

**Expected primary output:** `outputs/train_data.parquet`, `outputs/test_data.parquet`.

## 5. Gaussian Copula Synthesizer

**Functionality:** Generates synthetic tabular records using a statistical Gaussian Copula model.

**Docker image:** `ghcr.io/sydgep/docker-images/copula_synthesizer:latest`

**Configuration:** `container_input` (real training data); `container_output` (synthetic data).

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input=/app/outputs/train_data.parquet \
  -e container_output=/app/outputs/synthetic_data_copula.parquet \
  ghcr.io/sydgep/docker-images/copula_synthesizer:latest
```

**Expected primary output:** `outputs/synthetic_data_copula.parquet`.

## 6. CTGAN Synthesizer

**Functionality:** Generates synthetic tabular data with a conditional generative adversarial network; exposes training epochs and batch size.

**Docker image:** `ghcr.io/sydgep/docker-images/ctgan_synthesizer:latest`

**Configuration:** `container_input`; `container_output`; `epochs`; `batch_size`.

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input=/app/outputs/train_data.parquet \
  -e container_output=/app/outputs/synthetic_data_ctgan.parquet \
  -e epochs=300 \
  -e batch_size=100 \
  ghcr.io/sydgep/docker-images/ctgan_synthesizer:latest
```

**Expected primary output:** `outputs/synthetic_data_ctgan.parquet`.

## 7. CopulaGAN Synthesizer

**Functionality:** Generates synthetic data with a GAN-based method using copula transformations; exposes training epochs and batch size.

**Docker image:** `ghcr.io/sydgep/docker-images/copula_gan_synthesizer:latest`

**Configuration:** `container_input`; `container_output`; `epochs`; `batch_size`.

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input=/app/outputs/train_data.parquet \
  -e container_output=/app/outputs/synthetic_data_copula_gan.parquet \
  -e epochs=300 \
  -e batch_size=100 \
  ghcr.io/sydgep/docker-images/copula_gan_synthesizer:latest
```

**Expected primary output:** `outputs/synthetic_data_copula_gan.parquet`.

## 8. Statistical Fidelity Evaluation

**Functionality:** Compares real and synthetic data using distribution similarity (Jensen–Shannon-based) and mixed-type pairwise association similarity (Spearman, Cramér’s V, correlation ratio).

**Docker image:** `ghcr.io/sydgep/docker-images/statistical_evaluation:latest`

**Configuration:** `container_input_real`; `container_input_synth`; `container_output`.

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input_real=/app/outputs/train_data.parquet \
  -e container_input_synth=/app/outputs/synthetic_data_copula.parquet \
  -e container_output=/app/outputs/statistical_evaluation_results_copula.parquet \
  ghcr.io/sydgep/docker-images/statistical_evaluation:latest
```

**Expected primary output:** `outputs/statistical_evaluation_results_copula.parquet`.

## 9. Machine Learning Utility Evaluation

**Functionality:** Compares models trained on real data (TRTR) and synthetic data (TSTR), evaluating both on the same held-out real test set. Supports classification and regression models.

**Docker image:** `ghcr.io/sydgep/docker-images/ml_utility_evaluation:latest`

**Configuration:** `container_input_real_train`; `container_input_real_test`; `container_input_synth`; `container_output`; `feature_columns`; `target_col`; `task_type`; `model_name`; `metrics`.

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input_real_train=/app/outputs/train_data.parquet \
  -e container_input_real_test=/app/outputs/test_data.parquet \
  -e container_input_synth=/app/outputs/synthetic_data_copula.parquet \
  -e container_output=/app/outputs/utility_evaluation_results_copula.parquet \
  -e "feature_columns=Age, Gender, Marital_Status, Diagnosis, Treatment" \
  -e target_col=Outcome \
  -e task_type=classification \
  -e model_name=GradientBoostingClassifier \
  -e "metrics=accuracy, f1, roc_auc" \
  ghcr.io/sydgep/docker-images/ml_utility_evaluation:latest
```

**Expected primary output:** `outputs/utility_evaluation_results_copula.parquet`.

## 10. Privacy Attack Evaluation

**Functionality:** Estimates disclosure risks using singling-out, linkability, and inference attacks, including baseline/control comparisons. Results indicate estimated risk, not a guarantee of anonymity.

**Docker image:** `ghcr.io/sydgep/docker-images/privacy_attack_evaluation:latest`

**Configuration:** `container_input_real_train`; `container_input_real_test`; `container_input_synth`; `container_output`; `quasi_identifiers`; `sensitive_attributes`; `linkability_aux_cols_1`; `linkability_aux_cols_2` (as named in the supplied workflow).

**Run with Docker:**

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs:rw" \
  -e container_input_real_train=/app/outputs/train_data.parquet \
  -e container_input_real_test=/app/outputs/test_data.parquet \
  -e container_input_synth=/app/outputs/synthetic_data_copula.parquet \
  -e container_output=/app/outputs/privacy_evaluation_results_copula.parquet \
  -e "quasi_identifiers=Age, Gender, Marital_Status, Occupation, Admission_Date" \
  -e "sensitive_attributes=Diagnosis, Treatment, Outcome" \
  -e "linkability_aux_cols_1=Age, Gender, Occupation" \
  -e "linkability_aux_cols_2=Admission_Date, Diagnosis, Treatment" \
  ghcr.io/sydgep/docker-images/privacy_attack_evaluation:latest
```

**Expected primary output:** `outputs/privacy_evaluation_results_copula.parquet`.

## Notes on component interfaces

- All paths in `-e` arguments refer to **container paths**, not host paths. The `-v` option maps host files to those paths.
- Lists are passed as JSON where required by the component (for example, `identifier_columns`); other examples follow the comma-separated strings used in the reference workflow.
- **Privacy parameter names require confirmation:** the supplied workflow uses `linkability_aux_cols_1` and `linkability_aux_cols_2`, while an earlier image specification used `linkability_aux_cols1` and `linkability_aux_cols2`. Check the published container implementation before relying on the linkability configuration.
- The listed output filenames are the destinations requested in the reference workflow. Some images may also produce CSV summaries or figures.
- The components are independent; see the [Orchestration Guide](workflow-orchestration.md) for connecting them in a custom workflow.
