#  Hospital Emergency Room Performance & Prediction Project

**Author:** Farooq Shah  
**Organization:** DEVSIL (SMC-PRIVATE) LIMITED  
**Project Type:** Data Analysis + Machine Learning (Regression)  
**Dataset:** [ER Wait Time — Kaggle (rivalytics)](https://www.kaggle.com/datasets/rivalytics/er-wait-time)

---

##  Project Overview

This project analyzes Emergency Room (ER) performance data from 5 hospitals across 5,000 patient visits. The goal is to help hospital management:

- Understand **when and why** wait times are high
- Identify **which patients** are most affected
- **Predict Total Wait Time** using machine learning
- Provide **actionable recommendations** to reduce delays

---

##  Repository Structure

```
 ER_Prediction_Project/
├──  Hospital_ER_Performance_Prediction.ipynb   ← Main Jupyter Notebook (full analysis)
├──  er_wait_time.csv                           ← Dataset (5,000 ER visits)
├──  ER_Executive_Report.docx                   ← 2-Page Executive Summary Report
└──  README.md                                  ← This file
```

---

##  Dataset Description

| Column | Description |
|--------|-------------|
| `Visit ID` | Unique visit identifier |
| `Patient ID` | Unique patient identifier |
| `Hospital ID / Name` | One of 5 hospitals |
| `Region` | Urban or Rural |
| `Visit Date` | Datetime of ER arrival |
| `Day of Week` | Monday–Sunday |
| `Season` | Winter / Spring / Summer / Fall |
| `Time of Day` | Early Morning / Late Morning / Afternoon / Evening / Night |
| `Urgency Level` | Low / Medium / High / Critical |
| `Nurse-to-Patient Ratio` | Number of nurses per patient group |
| `Specialist Availability` | Number of specialists available |
| `Facility Size (Beds)` | Total hospital bed count |
| `Time to Registration (min)` | Minutes from arrival to registration |
| `Time to Triage (min)` | Minutes from registration to triage |
| `Time to Medical Professional (min)` | Minutes from triage to doctor contact |
| `Total Wait Time (min)` | **Target variable** — total ER wait in minutes |
| `Patient Outcome` | Discharged / Admitted / Left Without Being Seen |
| `Patient Satisfaction` | Rating 1 (lowest) to 5 (highest) |

---

##  Project Structure (Notebook Parts)

### Part 1 — Data Cleaning & Preparation
- Missing value check and handling
- Duplicate removal
- Datetime conversion
- New feature creation: Hour, Day, Month, Quarter, Is_Weekend
- Urgency & satisfaction grouping
- Outlier detection and Winsorization (IQR method)

### Part 2 — Exploratory Data Analysis (15+ Visualizations)
- Descriptive statistics: mean, median, variance, std dev
- Wait time distribution (histogram, KDE, pie chart)
- Patient volume by hour, weekday, season, month
- Wait time by urgency level, region, time of day (box plots, violin plots)
- Satisfaction and outcome analysis
- Staffing & hospital-level comparisons
- Heatmap: patients by day × time of day

### Part 3 — Statistical Reasoning
- Independent samples t-test: Urban vs Rural wait times
- Mean vs Median discussion (right-skewed distribution)
- Skewness and kurtosis analysis
- Q-Q normality plot
- Full correlation heatmap with interpretation

### Part 4 — Feature Engineering
| Feature | Reason |
|---------|--------|
| `Is_Weekend` | Weekend shifts have different staffing |
| `Is_Evening_Night` | Evening/Night are peak overload periods |
| `Staffing_Pressure` | Urgency ÷ nurse ratio — measures ER strain |
| `Specialist_Gap` | Urgency minus specialist count |
| `Pre_Care_Delay` | Registration + Triage time combined |
| Encoded categoricals | Required for ML models |

### Part 5 — Machine Learning Modeling
- **Problem type:** Regression (predicting `Total Wait Time (min)`)
- **Train/Test Split:** 80% / 20%
- **Models trained:**
  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. Gradient Boosting Regressor

### Part 6 — Model Evaluation
| Metric | Used For |
|--------|----------|
| MAE | Average absolute error in minutes |
| MSE | Penalises large prediction errors |
| RMSE | Root MSE — interpretable in minutes |
| R² | Proportion of variance explained |

Includes: model comparison bar charts, Actual vs Predicted scatter plot, Feature Importance chart.

### Part 7 — Business Recommendations
8 actionable recommendations covering:
- Staffing adjustments for evening/night shifts
- Fast-track lanes for low-urgency patients
- Registration digitization
- Rural telemedicine specialists
- ML-based proactive staffing
- Real-time patient flow dashboards
- Satisfaction improvement strategies
- Seasonal surge planning

---

## ⚙️ How to Run

### Requirements
```
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Run the notebook
```bash
jupyter notebook Hospital_ER_Performance_Prediction.ipynb
```

> Make sure `er_wait_time.csv` is in the **same folder** as the notebook.

---

##  Key Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | ~12–18 min | ~16–22 min | ~0.85–0.92 |
| Decision Tree | ~8–12 min | ~12–16 min | ~0.90–0.95 |
| **Random Forest** | **~6–10 min** | **~9–13 min** | **~0.95–0.98** |
| Gradient Boosting | ~6–10 min | ~9–13 min | ~0.95–0.98 |

> Actual values depend on train/test split. Run the notebook to see live results.

---

##  Best Model

**Random Forest** or **Gradient Boosting** — both capture non-linear relationships between staffing, urgency, timing, and wait time better than simpler models.

---

## 📞 Contact

**Farooq Shah**  
DEVSIL (SMC-PRIVATE) LIMITED  
Project: Phase 1 — Project 8
