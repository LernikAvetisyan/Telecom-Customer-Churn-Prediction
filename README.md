# Telecom-Customer-Churn-Prediction
COMP 442: Machine Learning

## Project Overview
Predict customer churn on the Telco dataset. We implemented Logistic Regression, Random Forest, and Histogram Gradient Boosting.  
**Chosen model:** Logistic Regression (best recall/accuracy trade-off for the business goal).

## Dataset
- Raw CSV: `data/Telco-Customer-Churn.csv`
- Preprocessed/engineered artifacts are written to `data/` and `artifacts/` when you run the script.

## Requirements
- Python 3.10+ (tested)
- See `requirements.txt`

## Setup (recommended)
```bash
# 1) Create & activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
# Activate venv first (see above), then:
python churn_predictor.py            # saves top-10 predicted churners to Example/
python churn_predictor.py 25         # top-25
python churn_predictor.py FULL       # all predicted churners
