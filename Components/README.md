# Components

This directory contains the independently executable components used by the Privacy-Preserving Framework.

Each component directory normally contains task_name.py and its corresponding Dockerfile.task

The Python file implements the component logic, while `Dockerfile.task` packages the component and its dependencies as a Docker image.

---

## 1. Building Docker Images

### Prerequisites

Ensure that Docker is installed and running:

```bash
docker --version
docker buildx version
docker info
```

Authenticate with the target container registry before pushing an image. For GitHub Container Registry:

```bash
echo "$CR_PAT" | docker login ghcr.io -u GITHUB_USERNAME --password-stdin
```

Replace:

* `CR_PAT` with a GitHub Personal Access Token containing the required package permissions;
* `GITHUB_USERNAME` with your GitHub username.

### Create a Buildx Builder

Create and activate a reusable Buildx builder:

```bash
docker buildx create \
  --name multiarch-builder \
  --driver docker-container \
  --use
```

Initialize it:

```bash
docker buildx inspect --bootstrap
```

An existing builder can be selected with:

```bash
docker buildx use multiarch-builder
```

### Build and Push a Multi-Platform Image

Navigate to the directory containing `task_name.py` and `Dockerfile.task`:

Build the image for AMD64 and ARM64:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.task \
  -t ghcr.io/insafedare/component-name:latest \
  --push .
```

Replace `component-name` with the name of the component being built.

For example:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.task \
  -t ghcr.io/insafedare/anonymization:latest \
  --push .
```

A versioned release can be published using more than one tag:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.task \
  -t ghcr.io/insafedare/anonymization:1.0.0 \
  -t ghcr.io/insafedare/anonymization:latest \
  --push .
```

Verify the published image:

```bash
docker buildx imagetools inspect \
  ghcr.io/insafedare/anonymization:latest
```

The output should list both:

```text
linux/amd64
linux/arm64
```

---

## 2. Component Overview

### Data Preparation

#### Data Ingestion

Reads selected columns from supported input formats and converts the data into a standard intermediate representation used by subsequent components.

#### Data Cleaning

Identifies or removes incomplete and invalid records, including missing values, empty strings, and configurable invalid-value markers.

---

### Privacy-Preserving Transformation

#### Direct Identifier Removal

Removes explicitly specified direct identifiers, such as patient names, record numbers, or other uniquely identifying attributes.

#### Anonymization

Transforms quasi-identifiers using techniques such as generalization, suppression, grouping, or value reduction to limit re-identification risk.

#### CTGAN Synthesizer

Generates synthetic tabular data using a Conditional Tabular Generative Adversarial Network.

#### VAE Synthesizer

Generates synthetic records using a Variational Autoencoder-based model.

#### Gaussian Copula Synthesizer

Produces synthetic tabular data by modelling column distributions and their statistical dependencies using a Gaussian copula.

---

### Privacy Evaluation

#### k-Anonymity

Measures the minimum number of records sharing the same combination of quasi-identifier values.

#### l-Diversity

Evaluates the diversity of sensitive-attribute values within groups of records sharing the same quasi-identifiers.

#### t-Closeness

Measures the difference between sensitive-attribute distributions within equivalence classes and the overall dataset distribution.

#### Attribute Inference Attack

Evaluates whether sensitive attributes can be inferred from the remaining available attributes.

#### Record Linkage Risk

Estimates the risk that privacy-preserving records can be linked to corresponding records in the original dataset.

#### Distance to Closest Record

Computes the similarity between generated records and their closest original records to identify possible memorization or excessive resemblance.

---

### Statistical Evaluation

#### Kolmogorov–Smirnov Evaluation

Compares numerical distributions between original and privacy-preserving datasets.

#### Wasserstein Distance

Measures the distance between numerical distributions in the original and transformed data.

#### Jensen–Shannon Divergence

Measures differences between categorical or discrete probability distributions.

---

### Machine Learning Utility

#### Utility Evaluation

Evaluates whether privacy-preserving datasets retain sufficient predictive information for downstream machine learning.

The component supports:

* classification;
* regression;
* `TrainReal_TestReal`;
* `TrainSynth_TestReal`;
* automatic preprocessing;
* feature encoding;
* missing-value handling;
* model training; and
* standardized performance evaluation.

Classification outputs may include accuracy, F1 score, and ROC-AUC. Regression outputs may include MAE, RMSE, and (R^2).

---


