
# Reproducible Synthetic Data Generation and Evaluation Pipelines

A modular and reproducible framework for constructing, evaluating, and sharing privacy-preserving healthcare datasets.

The framework provides an end-to-end workflow that combines healthcare data preparation, direct identifier removal, anonymization, synthetic data generation, privacy evaluation, statistical fidelity assessment, and machine learning utility evaluation. Each operation is implemented as an independent Docker container and can be executed individually or orchestrated into complete workflows using Prefect.

The primary objective of the framework is to facilitate reproducible research and trustworthy AI by providing reusable software components that enable researchers and practitioners to construct privacy-preserving healthcare datasets while preserving analytical value.

---

## Features

* Modular architecture
* A set of reusable components
* Docker-based task-containerization for reproducible execution
* Prefect workflow orchestration and visualization

---

## Current Components

These components can be executed independently or combined into complete end-to-end workflows.

### Data Preparation

* Data ingestion
* Data cleaning
* Data splitting
* De-identification

### Synthetic Data generation

* Gaussian Copula
* CTAGN
* Copula GAN)

### Privacy Evaluation

* Singling-out attack
* Linkability attack
* Inference attack
 

### Statistical Evaluation

* Jensen–Shannon distance
* Peer-wise Correlation

### Machine Learning Utility

Supports both classification and regression tasks through automated preprocessing and standardized evaluation workflows.

---

## Citation

If you use this framework or any of its components in your research, please cite the associated publication.

Citation information will be added upon publication.

---

## License

This project is licensed under the Apache License 2.0.

See the LICENSE file for details.

---

## Acknowledgements

This work has been developed within the context of INSAFEDARE project
