# Telecom Customer Churn Prediction

> Machine learning pipeline for predicting telecom customer churn using Python, pandas, scikit-learn, feature engineering, cross-validation, and threshold tuning.

> COMP 442: Machine Learning - Summer 2025 university project.

![Python](https://img.shields.io/badge/Language-Python-blue)
![pandas](https://img.shields.io/badge/Data-pandas-purple)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Model](https://img.shields.io/badge/Models-LR%20%7C%20RF%20%7C%20HGB-green)
![Course](https://img.shields.io/badge/Course-COMP%20442-lightgrey)

---

## About

This repository contains a machine learning project for **COMP 442: Machine Learning**. The goal is to predict which telecom customers are most likely to churn so that a company can identify at-risk customers early and take retention actions such as promotions, contract adjustments, or personalized support.

The project uses the public **Telco Customer Churn** dataset and builds an end-to-end machine learning workflow including data cleaning, feature engineering, model training, cross-validation, threshold selection, model comparison, and prediction export.

This was submitted as a university group project. My contribution included the core implementation, testing, documentation, report preparation, and final project coordination.

---

## Business Objective

Customer churn is a major issue in the telecom industry because losing existing customers creates revenue loss and additional acquisition costs. The business goal is to identify customers who are likely to leave before they actually churn.

The project was designed around two main success criteria:

| Criterion | Target | Purpose |
|---|---:|---|
| Recall | At least 80% | Catch most actual churners so the business can intervene |
| Accuracy | At least 70% in final evaluation | Keep overall predictions useful while prioritizing churn detection |

The project originally considered a higher accuracy target, but the final evaluation showed that prioritizing recall is more appropriate for this imbalanced business problem.

---

## Dataset

The project uses the **Telco Customer Churn** dataset.

Expected file path:

```text
data/Telco-Customer-Churn.csv
```

The dataset contains customer-level telecom information such as demographics, account details, services, contract type, payment method, charges, tenure, and churn status.

| Dataset Area | Examples |
|---|---|
| Customer demographics | gender, SeniorCitizen, Partner, Dependents |
| Account information | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV |
| Billing | MonthlyCharges, TotalCharges |
| Target variable | Churn |

For this portfolio version, the CSV is included so the project can run immediately after cloning.

---

## Project Workflow

The workflow follows the machine learning project lifecycle used in the course:

| Phase | Description |
|---|---|
| Business Understanding | Define churn prediction as a recall-focused retention problem |
| Data Understanding | Inspect telecom customer variables and churn imbalance |
| Data Preparation | Clean missing values, convert data types, cap outliers, and prepare features |
| Feature Engineering | Create business-driven features and encode categorical variables |
| Modeling | Train Logistic Regression, Random Forest, and Histogram Gradient Boosting |
| Evaluation | Compare models using recall, precision, F1, accuracy, ROC-AUC, and PR-AUC |
| Output Export | Save model artifacts and predicted churn customer lists |

---

## Data Cleaning

The script performs several cleaning steps before modeling:

| Step | Description |
|---|---|
| Convert `TotalCharges` | Converts text values to numeric values |
| Handle blank charges | Replaces invalid `TotalCharges` values using `MonthlyCharges * tenure` |
| Fill missing values | Uses median for numeric columns and mode for categorical columns |
| Handle outliers | Applies IQR-based clipping to selected numeric fields |
| Encode target | Converts `Churn` from `Yes/No` into `1/0` |

These steps make the dataset consistent, numeric, and ready for machine learning.

---

## Feature Engineering

The final modeling table contains **32 input features**.

| Feature Type | Count | Examples |
|---|---:|---|
| Numeric / binary features | 7 | `tenure`, `MonthlyCharges`, `avg_spend_per_month`, `num_services`, `is_long_contract` |
| One-hot encoded categorical features | 25 | `Contract_Two year`, `InternetService_Fiber optic`, `PaymentMethod_Electronic check` |

Important engineered features include:

| Feature | Purpose |
|---|---|
| `avg_spend_per_month` | Estimates average customer spending over time |
| `is_long_contract` | Flags customers with one-year or two-year contracts |
| `is_electronic_check` | Flags a payment method often associated with churn risk |
| `num_services` | Counts how many add-on services a customer uses |
| `tenure_group` | Groups tenure into interpretable ranges |
| `monthly_tier` | Groups monthly charges into low, medium, and high tiers |

Categorical fields are converted using one-hot encoding so the models can use them as numerical inputs.

---

## Models Compared

The project compares three classification models:

| Model | Reason Used |
|---|---|
| Logistic Regression | Interpretable baseline with stable probability outputs |
| Random Forest | Captures non-linear relationships and feature interactions |
| Histogram Gradient Boosting | Efficient boosted-tree model for structured/tabular data |

Class imbalance was handled using class weights where supported and threshold tuning for all models.

---

## Threshold Tuning Strategy

Instead of using the default `0.50` decision threshold, the project uses out-of-fold probabilities from stratified cross-validation to select a better threshold.

The threshold selection process:

1. Train models with stratified 5-fold cross-validation.
2. Generate out-of-fold churn probabilities.
3. Sweep thresholds from 0.10 to 0.90.
4. Keep thresholds that satisfy recall ≥ 80%.
5. Choose the threshold with the best F1 score among feasible options.
6. Evaluate final performance on the held-out test set.

This strategy matches the business goal because missing churners is more costly than contacting some customers who may not churn.

---

## Final Results

All three models satisfied the final recall and accuracy success criteria.

| Model | Threshold | Recall | Accuracy | Notes |
|---|---:|---:|---:|---|
| Logistic Regression | 0.45 | 82.9% | 72.4% | Selected final model |
| Random Forest | 0.30 | 82.9% | 72.3% | Similar recall, less interpretable |
| Histogram Gradient Boosting | 0.25 | 81.3% | 74.4% | Highest accuracy, slightly lower recall |

Although Histogram Gradient Boosting achieved slightly higher accuracy, **Logistic Regression** was selected as the final model because it met the recall target, maintained acceptable accuracy, and offered better interpretability for business stakeholders.

---

## Why Recall Matters More Than Accuracy

The churn class is imbalanced. Most customers do not churn, so a model can appear accurate by predicting that most customers will stay. However, that would fail the business goal because it would miss many customers who are actually at risk.

For this project:

- Recall is the primary metric because the business wants to catch churners early.
- Accuracy is still tracked, but it is not allowed to override the recall objective.
- A high-accuracy model that misses too many churners is not useful for retention planning.

This is why the final results prioritize recall around 80%+ even though accuracy remains in the low 70% range.

---

## Project Architecture

```text
Telecom-Customer-Churn-Prediction/
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── README.md
├── docs/
│   ├── Final_Project_Report.pdf
│   └── Final_Project_Presentation.pptx
├── churn_predictor.py
├── README.md
├── requirements.txt
└── .gitignore
```

Generated files are created when the script runs and are ignored by Git:

```text
artifacts/
Example/
outputs/
data/telco_preprocessed_for_model.csv
data/telco_preprocessed_with_normalized.csv
data/feature_names_after_encoding.txt
data/feature_card.json
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Models | Logistic Regression, Random Forest, Histogram Gradient Boosting |
| Evaluation | Recall, Precision, F1, Accuracy, ROC-AUC, PR-AUC |
| Serialization | joblib |
| Output Files | CSV, XLSX |
| Development | VS Code |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/LernikAvetisyan/Telecom-Customer-Churn-Prediction.git
cd Telecom-Customer-Churn-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the churn prediction pipeline

```bash
python churn_predictor.py
```

This runs the full workflow and exports the top 10 predicted churn customers.

---

## Run Options

The script supports different output sizes for predicted churn customers.

```bash
python churn_predictor.py
```

Exports the top 10 predicted churners.

```bash
python churn_predictor.py 25
```

Exports the top 25 predicted churners.

```bash
python churn_predictor.py FULL
```

Exports all predicted churners.

Generated prediction files are written to:

```text
Example/
```

---

## Outputs

When the script runs, it may generate:

| Output | Purpose |
|---|---|
| `artifacts/model.pkl` | Saved trained model |
| `artifacts/inference_config.json` | Model threshold and feature contract |
| `artifacts/results_table.csv` | Model comparison results |
| `artifacts/threshold_sweeps_*.csv` | Threshold sweep metrics |
| `Example/churn_predictions_top10.csv` | Top predicted churn customers |
| `Example/churn_predictions_top10.xlsx` | Excel version of predictions |

These files are generated automatically and are not committed to GitHub.

---

## Key Skills Demonstrated

- Data cleaning and preprocessing
- Handling missing values and text-to-numeric conversion
- Feature engineering from business logic
- One-hot encoding for categorical variables
- Binary classification
- Class imbalance handling
- Stratified train/test splitting
- Stratified cross-validation
- Out-of-fold probability threshold tuning
- Model comparison and selection
- Business-focused metric prioritization
- Exporting prediction results for review

---

## Documents

The final course documents are stored in the `docs/` folder:

| File | Description |
|---|---|
| `Final_Project_Report.pdf` | Final written report with methodology, analysis, results, and conclusion |
| `Final_Project_Presentation.pptx` | Final project presentation slides |

---

## Limitations and Future Work

This project was completed as a university machine learning project. Future improvements could include:

- Moving preprocessing into a full scikit-learn `Pipeline` or `ColumnTransformer`
- Testing additional models such as XGBoost, LightGBM, or CatBoost
- Applying balancing methods such as SMOTE or undersampling
- Adding hyperparameter search with GridSearchCV or Bayesian optimization
- Building a small web interface for churn-risk scoring
- Preserving `customerID` in prediction exports for operational use
- Adding automated tests for preprocessing and inference consistency

---

## Resume Summary

Built a telecom customer churn prediction pipeline for a COMP 442 Machine Learning project using Python, pandas, and scikit-learn. Implemented data cleaning, feature engineering, one-hot encoding, stratified cross-validation, threshold tuning, and model comparison across Logistic Regression, Random Forest, and Histogram Gradient Boosting. Selected Logistic Regression as the final model based on recall-first business criteria, interpretability, and stable performance.

---

## Disclaimer

This project is an educational machine learning project using a public telecom churn dataset. It is not connected to a real telecom company and should not be treated as a production customer-retention system.
