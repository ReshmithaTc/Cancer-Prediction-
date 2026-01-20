# 🧠 Cancer Risk Prediction System (Machine Learning Pipeline)

This project implements an **end-to-end Machine Learning pipeline** to predict **high cancer risk vs low cancer risk** using patient health data.  
It covers **data preprocessing, feature engineering, multi-model training, evaluation, model selection, and inference**.

---

## 📌 Project Overview

- **Problem Type:** Binary Classification (High Risk / Low Risk)
- **Target Variable:** `High_Risk` (derived from `Risk_Score`)
- **Dataset:** `AI_Brosnan_CancerDataset.xlsx`
- **Best Model Selection:** Based on cross-validation accuracy
- **Final Output:** Risk prediction with probability for new patients

---


## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:**
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `pickle`

---

## 📊 Features Used

### 🔢 Numerical Features
- Age, Height, Weight, BMI
- Blood Pressure (Systolic, Diastolic)
- Heart Rate, Temperature
- Blood Sugar, Cholesterol, Hemoglobin
- Exercise Hours / Week
- Hospital Visits / Year

### 🧠 Engineered Features
- `BMI_Age_Interaction`
- `BP_Ratio`
- `Health_Score`
- `High_Cholesterol`
- `High_Blood_Sugar`
- `Low_Exercise`

### 🔤 Categorical Encoding
- **Label Encoding (Ordinal):**
  - Alcohol_Consumption
  - Insurance_Type
- **One-Hot Encoding (Nominal):**
  - Gender
  - Smoking
  - Diabetes
  - Hypertension
  - Heart_Disease

---

## 🔄 Pipeline Flow

1. **Data Loading**
2. **Feature Engineering**
3. **Encoding (Label + One-Hot)**
4. **Train-Test Split (80–20, Stratified)**
5. **Standard Scaling**
6. **Multi-Model Training**
7. **Cross-Validation**
8. **Best Model Selection**
9. **Evaluation & Visualization**
10. **Inference on New Patient Data**

---

## 🤖 Models Trained

| Model | Description |
|------|------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Rule-based learning |
| Random Forest | Ensemble tree model |
| Support Vector Machine | Margin-based classifier |
| K-Nearest Neighbors | Distance-based model |

✅ **Best model is automatically selected** using mean cross-validation accuracy.

---

## 📈 Model Evaluation

Metrics calculated for each model:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### 📊 Outputs Generated
- `model_evaluation_dashboard.png`
- Confusion matrices (per model)
- `model_evaluation_summary.csv`




