# Advanced Social Media Engagement Prediction
### Phase 1 — Project 9 | Devsil

---

## Overview

This project builds a complete machine learning pipeline to predict social media engagement using a dataset of 12,000 posts across Instagram, Twitter, Reddit, and YouTube. It covers the full data science workflow — from raw data exploration through to a tuned ensemble model — applied to 28 features including post content, sentiment, campaign context, and user history.

---

## Project Structure

```
├── Social_Media_Engagement_Prediction.ipynb   # Main notebook (all tasks)
├── Social_Media_Engagement_Dataset.csv        # Raw dataset (12,000 rows)
├── Project_Report.docx                        # Full written report
└── README.md                                  # This file
```

---

## Dataset

| Category | Columns |
|---|---|
| Post metadata | `post_id`, `timestamp`, `day_of_week`, `platform` |
| User info | `user_id`, `location`, `language` |
| Content | `text_content`, `hashtags`, `mentions`, `keywords`, `topic_category` |
| Sentiment & Emotion | `sentiment_score`, `sentiment_label`, `emotion_type`, `toxicity_score` |
| Engagement metrics | `likes_count`, `shares_count`, `comments_count`, `impressions`, `engagement_rate` |
| Campaign | `brand_name`, `product_name`, `campaign_name`, `campaign_phase` |
| User history | `user_past_sentiment_avg`, `user_engagement_growth`, `buzz_change_rate` |

---

## Tasks Completed

### 1. Data Understanding & Exploration
- Loaded dataset with pandas, inspected shape, dtypes, and missing values
- Generated descriptive statistics for all numeric features
- Plotted distributions of all five engagement metrics

### 2. Exploratory Data Analysis (EDA)
- Engagement rate distribution (histogram + KDE)
- Average engagement by platform
- Average engagement by campaign phase
- Correlation heatmap of numeric variables
- Sentiment vs engagement rate (boxplot)
- Average engagement by day of week

### 3. Data Preprocessing
- Median imputation for numeric columns, mode imputation for categorical
- Outlier removal using the IQR method on `engagement_rate`
- Label encoding for 7 categorical columns
- StandardScaler normalization before model training

### 4. Feature Engineering
- `total_engagement` = likes + shares + comments
- `hour`, `month`, `is_weekend` from timestamp
- `hashtag_count`, `mention_count` from post text
- `text_length` character count
- `sentiment_x_growth` interaction feature

### 5. Machine Learning Models
Trained and compared 7 models on binary engagement classification (High vs Low):
- Linear Regression (RMSE)
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Naive Bayes
- Support Vector Machine (RBF kernel)

### 6. Ensemble & Advanced Models
- Random Forest (200 trees)
- XGBoost
- CatBoost
- Gradient Boosting
- AdaBoost
- Feature importance plot from XGBoost

### 7. Model Optimization
- GridSearchCV hyperparameter tuning on Random Forest
- scikit-learn Pipeline (scaler + classifier)
- 5-fold cross-validation
- Confusion matrix and full classification report

### 8. Unsupervised Learning — Clustering
- Elbow method + silhouette scores to find optimal k
- K-Means, Hierarchical, DBSCAN, and Gaussian Mixture Model clustering
- 2x2 cluster visualization panel

### 9. Dimensionality Reduction
- PCA with cumulative explained variance plot
- PCA 2D scatter colored by cluster
- Truncated SVD
- t-SNE visualization on a 2,000-sample subset

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost
```

### Steps

1. Place `Social_Media_Engagement_Dataset.csv` in the same folder as the notebook
2. Open `Social_Media_Engagement_Prediction.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells top to bottom (Kernel > Restart & Run All)

Python 3.9+ recommended.

---

## Key Results

| Model | Accuracy |
|---|---|
| XGBoost | ~0.99 |
| CatBoost | ~0.99 |
| Random Forest (tuned) | ~0.99 |
| Gradient Boosting | ~0.98 |
| Decision Tree | ~0.97 |
| KNN | ~0.94 |
| Logistic Regression | ~0.89 |
| Naive Bayes | ~0.72 |

---

## Technologies Used

- Python 3.12, pandas, NumPy
- matplotlib, seaborn
- scikit-learn — preprocessing, ML models, pipelines, clustering, PCA, t-SNE
- XGBoost, CatBoost
- Jupyter Notebook

---

## Author

Submitted as part of Phase 1 — Project 9 | Devsil Machine Learning Program
