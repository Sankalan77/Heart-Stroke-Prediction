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

Raw Dataset (stroke_dataset_smote.csv)<br>
│<br>
▼<br>
Data Loading & EDA<br>
│<br>
▼<br>
Feature / Target Split<br>
│<br>
▼<br>
Train-Test Split (80/20, stratified)<br>
│<br>
▼<br>
StandardScaler (Feature Normalization)<br>
│<br>
├──────────────────────────────────────┐<br>
▼                                      ▼<br>
Classical ML Models (×10)         Hybrid Stacking Models (×10)<br>
│                                      │<br>
▼                                      ▼<br>
Evaluation & Metrics              Evaluation & AUC-ROC<br>
│                                      │<br>
└──────────────┬───────────────────────┘<br>
▼<br>
Model Comparison & Selection<br>
│<br>
▼<br>
New Patient Prediction Demo

---

## 🤖 Models Evaluated

### Classical ML Classifiers

| # | Model | Key Config |
|---|---|---|
| 1 | **Logistic Regression** | `max_iter=1000` |
| 2 | **Decision Tree** | `random_state=42` |
| 3 | **Random Forest** | `n_estimators=100` |
| 4 | **K-Nearest Neighbors** | `n_neighbors=5` |
| 5 | **Support Vector Machine** | `kernel='rbf'` |
| 6 | **Naive Bayes** | Gaussian NB |
| 7 | **AdaBoost** | `n_estimators=100` |
| 8 | **Gradient Boosting** | `random_state=42` |
| 9 | **Extra Trees** | `n_estimators=100` |
| 10 | **XGBoost** | `eval_metric='logloss'` |

---

## 🧩 Hybrid Stacking Ensembles

10 stacking models were built using **3-model combinations** of the top base learners (ET, RF, XGB, KNN, GB), each stacked with a **Logistic Regression meta-classifier**.

| Hybrid Model | Base Learners |
|---|---|
| Hybrid_1_ET_RF_XGB | Extra Trees + Random Forest + XGBoost |
| Hybrid_2_ET_RF_KNN | Extra Trees + Random Forest + KNN |
| Hybrid_3_ET_XGB_KNN | Extra Trees + XGBoost + KNN |
| Hybrid_4_RF_XGB_KNN | Random Forest + XGBoost + KNN |
| Hybrid_5_ET_RF_GB | Extra Trees + Random Forest + Gradient Boosting |
| Hybrid_6_ET_XGB_GB | Extra Trees + XGBoost + Gradient Boosting |
| Hybrid_7_RF_XGB_GB | Random Forest + XGBoost + Gradient Boosting |
| Hybrid_8_ET_KNN_GB | Extra Trees + KNN + Gradient Boosting |
| Hybrid_9_RF_KNN_GB | Random Forest + KNN + Gradient Boosting |
| Hybrid_10_ET_RF_XGB | Extra Trees + Random Forest + XGBoost *(alt config)* |

> Meta-classifier: **Logistic Regression** (`max_iter=5000`, `cv=3–5`, `passthrough=True`)

---

## 📏 Evaluation Metrics

Each model is evaluated on:

- **Accuracy** — Overall correct predictions
- **Precision** — True positives out of all positive predictions
- **Recall** — True positives out of all actual positives
- **F1-Score** — Harmonic mean of Precision and Recall
- **AUC-ROC** — Area under the Receiver Operating Characteristic curve
- **Confusion Matrix** — TP / TN / FP / FN breakdown

> ⚠️ **Acceptance Threshold**: Models with accuracy **≥ 85%** are marked `ACCEPTED`; others are marked `REJECTED`.

---

## 📊 Visualizations

The project generates the following plots:

- 📊 Vertical bar chart — Accuracy comparison of all ML models
- 📊 Horizontal bar chart — Accuracy comparison of all ML models
- 🟥 Confusion matrices — For each individual model (heatmap)
- 📉 AUC-ROC curves — All classical models on one plot
- 📉 AUC-ROC curves — All hybrid stacking models on one plot

---

## 🏥 Patient Prediction Demo

The project includes a live prediction block for a **new patient**. The example uses an extreme high-risk profile:

```python
new_patient = {
    'Age': 78,
    'Hypertension': 1,
    'Heart_Disease': 1,
    'Average_Glucose_Level': 285.5,
    'BMI': 38.2,
    'Stress_Level': 9.5,
    'Blood_Pressure': 210,
    'Cholesterol': 320,
    'MRI_Result': 9.1,
    'Smoking_Status_Formerly': 1,
    'Family_History_Yes': 1,
    # ... (all features)
}
```

**Output example:**

========== BASIC MODELS ==========<br>
Random Forest      --> STROKE<br>
XGBoost            --> STROKE<br>
Extra Trees        --> STROKE<br>

========== HYBRID STACKING MODELS ==========<br>
Hybrid_1_ET_RF_XGB --> STROKE<br>
Hybrid_2_ET_RF_KNN --> STROKE

