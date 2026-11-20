# Datasets

This directory contains the datasets used to demonstrate and evaluate the Privacy-Preserving Framework.

The framework has been evaluated using three healthcare datasets of increasing complexity:

1. **Artificial Healthcare Dataset** (`data.csv`)
2. **Diabetes Dataset** (`diabetes.csv`)
3. **MIMIC-IV Dataset (Not Distributed)**

The first two datasets are included in this repository for testing and demonstration purposes. The MIMIC-IV dataset is **not included** because it is distributed under a Data Use Agreement (DUA).

---

# 1. Artificial Healthcare Dataset

### Description

It contains representative healthcare attributes including direct identifiers, quasi-identifiers, and sensitive attributes, allowing users to execute the complete framework without requiring access to real clinical data.

Typical use cases include:

* testing Docker components
* validating workflow execution
* evaluating synthetic data generators
* reproducing the experiments presented in the paper

---

# 2. Diabetes Dataset

### Source

Kaggle Breast Cancer Dataset

The dataset contains breast cancer patient information including demographic characteristics, clinical measurements, tumour information, treatment variables, and patient outcomes.

Within this framework, the dataset is used to demonstrate:

* direct identifier removal
* synthetic data generation
* privacy evaluation
* statistical fidelity assessment
* machine learning utility evaluation

---

# 3. MIMIC-IV Dataset

## Availability

The MIMIC-IV dataset is **not distributed** with this repository.

Researchers wishing to reproduce the experiments must obtain access directly from the official MIMIC-IV repository after completing the required training and Data Use Agreement.

Official website:

https://physionet.org/content/mimiciv/

---

## Dataset Used in This Work

The experiments reported in the accompanying paper use an integrated subset of the MIMIC-IV v2.0 database.

The integrated dataset was constructed from multiple MIMIC-IV tables by selecting patients diagnosed with **Heart Failure** using **ICD-9 codes I50 and 428**. Data integration and cohort construction are considered preprocessing steps and are outside the scope of this framework.

The resulting dataset contains:

* **64,303 patient admissions**
* **19 variables**

### Variables

| Variable               | Description                                    |
| ---------------------- | ---------------------------------------------- |
| hadm_id                | Hospital admission identifier                  |
| subject_id             | Patient identifier (pseudonymized in MIMIC-IV) |
| admittime              | Hospital admission date and time               |
| admission_type         | Admission category                             |
| admission_location     | Admission source                               |
| insurance              | Insurance type                                 |
| language               | Patient language                               |
| marital_status         | Marital status                                 |
| race                   | Patient race/ethnicity                         |
| gender                 | Patient gender                                 |
| anchor_age             | Patient age                                    |
| anchor_year            | Reference year used by MIMIC-IV                |
| anchor_year_group      | Anchor year interval                           |
| n_diagnoses            | Number of diagnosis records                    |
| n_prescription_records | Number of prescription records                 |
| systolic_bp            | Systolic blood pressure                        |
| diastolic_bp           | Diastolic blood pressure                       |
| BMI                    | Body Mass Index                                |
| hospital_expire_flag   | In-hospital mortality outcome                  |

This integrated dataset serves as an example input for evaluating privacy-preserving workflows and is **not redistributed** as part of this repository.

---


