# Week 5 — Advanced Machine Learning: House Price Prediction

**AnalystLab Africa Data Science Internship (Batch B)**

## Overview
This project explores a full regression modeling pipeline: baseline
Linear Regression, Decision Tree, Random Forest, and Gradient Boosting, with
hyperparameter tuning via GridSearchCV (Random Forest) and RandomizedSearchCV (Gradient
Boosting). All models are trained on 'log1p(price)', and predictions are back-transformed
with 'expm1' before scoring — so every metric below is in real currency units, not
log-price units.

## Dataset
- **Source:** Housing Prices Dataset (545 rows, 13 columns)
- **Target:** `price`
- **Features:** area, bedrooms, bathrooms, stories, parking, mainroad, guestroom,
  basement, hotwaterheating, airconditioning, prefarea, furnishingstatus
- Dataset link:  (included in this folder)

## Models & Results

| Model | R² | MAE (₦) | RMSE (₦) |
|---|---|---|---|
| **Gradient Boosting (Tuned — RandomizedSearchCV) — Best** | **0.6443** | **954,461** | **1,340,922** |
| Linear Regression (Baseline) | 0.6426 | 1,002,010 | 1,344,038 |
| Random Forest (default) | 0.6244 | 958,271 | 1,377,946 |
| Random Forest (Tuned — GridSearchCV) | 0.6188 | 966,126 | 1,388,175 |
| Gradient Boosting (default) | 0.6089 | 985,966 | 1,406,081 |
| Decision Tree (max_depth=5) | 0.5652 | 1,041,560 | 1,482,450 |

## Key Insights
- Tuning had **opposite effects** on the two ensembles: it improved Gradient Boosting
  (+0.035 R²) but slightly hurt Random Forest (−0.006 R²) — hyperparameter search needs
  to be validated per-algorithm, not assumed to help uniformly.
- The Linear Regression baseline was a surprisingly strong performer, beating three of
  the five more complex models — the price relationship is substantially linear once
  log-transforms and ordinal encodings are applied.
- A single Decision Tree was the weakest model, consistent with its higher variance on a
  dataset this size (545 rows).

## Files
- `housing_dataset-1.ipynb` — full notebook: preprocessing, EDA, model building, tuning,
  evaluation, visualizations
- `Week5_Performance_Comparison_Report_v2.docx`(named as summary_insights) — performance comparison report
- `Housing.csv` — dataset
- `README.md` — this file

## Tools
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook
