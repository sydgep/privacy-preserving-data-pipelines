
# Privacy-Preserving Framework

A modular and reproducible framework for constructing, evaluating, and sharing privacy-preserving healthcare datasets.

The framework provides an end-to-end workflow that combines healthcare data preparation, direct identifier removal, anonymization, synthetic data generation, privacy evaluation, statistical fidelity assessment, and machine learning utility evaluation. Each operation is implemented as an independent Docker container and can be executed individually or orchestrated into complete workflows using Prefect.

The primary objective of the framework is to facilitate reproducible research and trustworthy AI by providing reusable software components that enable researchers and practitioners to construct privacy-preserving healthcare datasets while preserving analytical value.

---

## Features

* Modular architecture with independently executable components
* Docker-based implementation for reproducible execution
* Prefect workflow orchestration
* Direct identifier removal
* Healthcare data anonymization
* Synthetic data generation using CTGAN, VAE and Gaussian Copula
* Privacy evaluation
  * k-anonymity
  * l-diversity
  * t-closeness
  * Attribute inference attack
  * Record linkage risk
  * Distance-to-Closest Record (DCR)
* Statistical fidelity evaluation

  * Kolmogorov–Smirnov statistic
  * Wasserstein distance
  * Jensen–Shannon divergence
* Machine learning utility evaluation

  * Classification and regression support
  * TrainReal_TestReal
  * TrainSynth_TestReal
* Reproducible workflows for multiple healthcare datasets

---

## Repository Structure

```text
privacy-preserving-framework/
├── components/             # Codes and Dockerfiles
├── workflows/              # Prefect workflows
├── datasets/               # Test datasets
├── README.md
├── LICENSE
├── CITATION.cff
└── CONTRIBUTING.md
```

---

## Current Components

These components can be executed independently or combined into complete end-to-end workflows.

### Data Preparation

* Data ingestion
* Data cleaning

### Privacy-Preserving Transformation

* Direct identifier removal
* Anonymization
* Synthetic data generators (CTGAN, VAE, Copula)

### Privacy Evaluation

* k-anonymity
* l-diversity
* t-closeness
* Attribute inference attack
* Record linkage risk
* Distance-to-Closest Record

### Statistical Evaluation

* Kolmogorov–Smirnov statistic
* Wasserstein distance
* Jensen–Shannon divergence

### Machine Learning Utility

Supports both classification and regression tasks through automated preprocessing and standardized evaluation workflows.

---

## Citation

If you use this framework in your research, please cite the associated publication.

Citation information will be added upon publication.

---

## License

This project is licensed under the Apache License 2.0.

See the LICENSE file for details.

---

## Acknowledgements

This work has been developed within the context of research on reproducible privacy-preserving healthcare data processing and trustworthy artificial intelligence.
