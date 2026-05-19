# Heart-Stroke-Prediction
# 🧠 Heart & Stroke Disease Prediction using Hybrid Stacking Ensemble

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Boosting-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SMOTE-Balanced-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-yellow?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</p>

<p align="center">
  <b>A comprehensive machine learning project that benchmarks 10 classical ML models and 10 hybrid stacking ensembles for early stroke prediction — achieving high accuracy on a SMOTE-balanced dataset.</b>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Project Pipeline](#-project-pipeline)
- [Models Evaluated](#-models-evaluated)
- [Hybrid Stacking Ensembles](#-hybrid-stacking-ensembles)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualizations](#-visualizations)
- [Patient Prediction Demo](#-patient-prediction-demo)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Results Summary](#-results-summary)

---

## 🔍 Overview

Stroke is one of the leading causes of death and disability worldwide. Early detection can significantly improve outcomes. This project builds and compares **20 machine learning models** — 10 classical algorithms and 10 hybrid stacking ensembles — to predict whether a patient is at risk of a stroke.

Key highlights:
- ✅ **SMOTE-balanced dataset** to handle class imbalance
- ✅ **10 classical ML models** benchmarked head-to-head
- ✅ **10 hybrid stacking classifiers** using 3-model combinations
- ✅ **Full evaluation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- ✅ **Live patient prediction** with new input data

---

## 📂 Dataset

| Property | Details |
|---|---|
| **File** | `stroke_dataset_smote.csv` |
| **Samples** | ~41,000 records |
| **Balancing** | SMOTE (Synthetic Minority Oversampling Technique) |
| **Target** | `Stroke` (0 = No Stroke, 1 = Stroke) |

### 🧬 Input Features

| Feature | Description |
|---|---|
| `Age` | Patient's age |
| `Hypertension` | Hypertension presence (0/1) |
| `Heart_Disease` | Existing heart disease (0/1) |
| `Average_Glucose_Level` | Blood glucose level |
| `BMI` | Body Mass Index |
| `Physical_Activity` | Activity level score |
| `Alcohol_Intake` | Alcohol consumption units |
| `Stress_Level` | Self-reported stress (1–10) |
| `Blood_Pressure` | Systolic BP reading |
| `Cholesterol` | Cholesterol level |
| `MRI_Result` | Neuroimaging score |
| `Sex_Male` | Gender (encoded) |
| `Ever_Married_Yes` | Marital status (encoded) |
| `Work_Type_*` | Work type (one-hot encoded) |
| `Residence_Type_Urban` | Urban/Rural (encoded) |
| `Smoking_Status_*` | Smoking history (one-hot encoded) |
| `Family_History_Yes` | Family stroke history (encoded) |

---

## ✨ Features

- 🔄 Full **data preprocessing pipeline** (scaling, encoding, splitting)
- 📊 **EDA** with class distribution analysis
- 🤖 **10 classical ML classifiers** trained and compared
- 🧩 **10 hybrid stacking ensembles** with 3-model base combinations
- 📈 **Accuracy bar charts** (vertical & horizontal)
- 🟥 **Confusion matrices** for all models
- 📉 **AUC-ROC curves** for classical and hybrid models
- ✅ **Acceptance criteria** validation (≥ 85% accuracy threshold)
- 🏥 **New patient stroke risk prediction** demo

---

## 🔁 Project Pipeline
