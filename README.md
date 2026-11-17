# Synthetic Data Generation and Evaluation Pipelines

A modular and reproducible framework for generating and evaluating synthetic healthcare datasets.

The framework provides an end-to-end workflow encompassing data ingestion, preprocessing, de-identification, synthetic data generation, and multidimensional evaluation. Each operation is implemented as an independent Docker container, enabling reusable, portable, and reproducible execution. Individual components can be executed independently or integrated into complete data pipelines orchestrated using Prefect.

The primary objective is to support reproducible research and the development of trustworthy, privacy-preserving healthcare datasets by providing reusable software components for data processing, synthetic data generation, and systematic evaluation of statistical fidelity, machine learning utility, and privacy-related risks.

---

## Features

- **Modular architecture:** Independent and reusable components for building customizable data pipelines.
- **Task-level containerization:** Docker-based execution environments that promote portability, consistency, and reproducibility.
- **Workflow orchestration:** Prefect-based execution, monitoring, and visualization of end-to-end data pipelines.
- **Multiple generation methods:** Support for statistical, deep learning-based, and hybrid synthetic data generation techniques.
- **Multidimensional evaluation:** Integrated assessment of statistical fidelity, predictive utility, and privacy-related disclosure risks.
- **Extensibility:** A component-based design that facilitates the integration of additional privacy-preserving techniques, generation algorithms, and evaluation methods.

---

## Current Components

The framework provides reusable components that can be executed independently or combined into complete end-to-end workflows.

### 1. Data Preparation

- **Data Ingestion:** Extraction of tabular data from supported file formats, databases, and REST APIs.
- **Data Cleaning:** Identification and handling of missing values.
- **De-identification:** Removal of specified direct identifiers.
- **Train/Test Splitting:** Partitioning datasets into training and testing subsets, with optional stratification.

### 2. Synthetic Data Generation

The framework supports three complementary approaches to tabular synthetic data generation:

- **Gaussian Copula:** A statistical model-based approach that captures marginal distributions and dependencies between variables.
- **CTGAN:** A deep learning-based approach using conditional generative adversarial networks.
- **CopulaGAN:** A hybrid approach combining statistical transformations with GAN-based synthetic data generation.

### 3. Privacy Evaluation

Privacy-related disclosure risks are assessed using the Anonymeter framework through three attack-based evaluations:

- **Singling-out:** Assesses the possibility of isolating individual records.
- **Linkability:** Evaluates whether records can be linked using auxiliary information.
- **Inference:** Assesses the possibility of inferring sensitive attributes from available information.

The evaluation reports estimated privacy risks alongside attack success rates and baseline/control results.

### 4. Statistical Fidelity Evaluation

Statistical fidelity is assessed through two complementary measures:

- **Distribution Similarity:** Measures similarity between real and synthetic feature distributions using Jensen–Shannon distance.
- **Correlation Similarity:** Evaluates preservation of pairwise relationships using Spearman correlation, Cramér's V, and correlation ratio, depending on feature types.

Both measures are expressed as similarity scores, where values closer to 100% indicate greater statistical similarity.

### 5. Machine Learning Utility Evaluation

Evaluates the extent to which synthetic datasets preserve information relevant to downstream machine learning tasks.

The component supports classification and regression models and compares two experimental settings:

- **Train on Real, Test on Real (TRTR):** Establishes predictive performance using real training data and held-out real test data.
- **Train on Synthetic, Test on Real (TSTR):** Evaluates predictive performance when synthetic data replace real training data.

The evaluation reports predictive performance metrics and retained utility relative to the real-data baseline.

---

## Citation

If you use this framework or any of its components in your research, please cite the associated publication.

Citation information will be added upon publication.

---

## License

This project is licensed under the Apache License 2.0.

See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work has been developed within the context of the **INSAFEDARE** project, which aims to support high-assurance, data-driven decision-making in medical device regulation through privacy-preserving data technologies and associated evaluation methodologies.
