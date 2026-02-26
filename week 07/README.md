# Employee Attrition Prediction
IBM HR Analytics Dataset | Devsil Phase 1 — Project 7

---

## Overview

This project builds an end-to-end machine learning pipeline to predict employee attrition. The goal is to identify employees at risk of leaving before they resign, and to surface the key drivers of turnover so that HR can act on them.

---

## Project Structure

```
employee-attrition-prediction/
├── employee_attrition_analysis.ipynb    # Main notebook — full pipeline
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset
├── README.md
```

Outputs generated when you run the notebook:

```
best_model.pkl          # Saved best model
scaler.pkl              # Fitted StandardScaler
model_comparison.csv    # All model metrics
fig_01 to fig_09.png    # Charts
```

---

## Analysis Pipeline

| Section | Description |
|---------|-------------|
| 1 | Environment setup and data loading |
| 2 | Data cleaning — missing values, duplicates, encoding, outlier detection |
| 3 | Exploratory data analysis — 9 charts |
| 4 | Statistical testing — t-tests, chi-square, confidence intervals |
| 5 | Feature engineering, SMOTE, train/test split, scaling |
| 6 | Training and comparing 5 ML models |
| 7 | Evaluation — confusion matrices, ROC curves, feature importance |
| 8 | Business insights and HR recommendations |

---

## Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Fast, visualizable |
| Random Forest | Best for feature importance |
| SVM | Strong on high-dimensional data |
| KNN | Instance-based, distance-weighted |

Primary metric: **ROC-AUC** — robust for imbalanced classification.  
Secondary metric: **Recall** — a missed attrition (false negative) costs more than a false alarm.

---

## Dataset

1,470 employees, 35 features.  
Source: IBM HR Analytics, publicly available on Kaggle.  
The CSV is included in this repository.

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scipy statsmodels joblib
```

Open `employee_attrition_analysis.ipynb` in VS Code or Jupyter and run all cells.

---

## Key Findings

- Overtime is the single strongest predictor of attrition
- Employees who left earned significantly less than those who stayed (p < 0.001)
- Attrition is concentrated in the first three years of tenure
- Low job satisfaction and poor work-life balance are leading indicators of departure
- Sales Representatives and Laboratory Technicians have the highest attrition rates

---

## Author

Built as part of Devsil Phase 1 — Real-World Industry Practice Projects.  
Role: Junior Data Scientist | Department: HR Analytics
