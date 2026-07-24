# Customer Churn Prediction

A production-style machine learning pipeline that predicts which telecom customers are likely to churn, using cost-based decision thresholds to align model behavior with real business economics.

## Overview

This project trains and compares two classifiers (Logistic Regression and XGBoost) on the IBM Telco Customer Churn dataset, then selects a decision threshold that minimizes total business cost — not just a generic metric like accuracy or F1. The result: a model whose predictions are tuned to how expensive it actually is to lose a customer vs. wasting a retention offer on someone who was never going to leave.

## Dataset

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
7,043 customers, 19 features (contract type, tenure, monthly charges, internet service, payment method, etc.), binary churn label.

## Project Structure

customer_churn_prediction/
├── data/
│ ├── raw/ # original Kaggle CSV
│ └── processed/ # cleaned data, train/test splits, scaler
├── models/ # trained model + comparison metrics
├── src/
│ ├── data/make_dataset.py # load + clean raw data
│ ├── features/build_features.py # encode, scale, split
│ └── models/
│ ├── train_model.py # train + cost-optimal threshold tuning
│ └── predict_model.py # inference on new customers
├── tests/ # unit tests for data + feature logic
├── .github/workflows/ # CI: lint + test on every push
├── Dockerfile # containerized pipeline (Chainguard base)
└── configs/base.yaml # pipeline configuration
## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

## Usage

Run each pipeline stage in order:

```bash
# 1. Clean raw data
python src/data/make_dataset.py data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv data/processed/churn_cleaned.csv

# 2. Build features (encode, scale, split train/test)
python src/features/build_features.py data/processed/churn_cleaned.csv data/processed

# 3. Train models and select the cost-optimal threshold
python src/models/train_model.py data/processed models --cost-fn 500 --cost-fp 100

# 4. Predict on new customers
python src/models/predict_model.py data/raw/new_customers.csv data/processed/predictions.csv
```

`--cost-fn` and `--cost-fp` let you adjust the assumed dollar cost of a missed churner vs. a wasted retention offer — the threshold is re-optimized accordingly.

## Methodology

1. **Cleaning:** convert `TotalCharges` to numeric, drop rows with missing values, drop the customer ID, encode the target as binary.
2. **Feature engineering:** one-hot encode categorical variables, standardize numeric variables.
3. **Modeling:** train Logistic Regression and XGBoost, both with class weighting to address the ~27% churn / 73% no-churn imbalance.
4. **Threshold selection:** rather than using the default 0.5 cutoff, search thresholds and pick the one that **minimizes total expected cost** — `(false negatives × cost of losing a customer) + (false positives × cost of a wasted offer)`.
5. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, and total business cost on a held-out test set.

## Results

At default cost assumptions ($500 per missed churner, $100 per wasted offer):

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 (default) | 80.4% | 64.9% | 57.0% | 60.7% | 0.836 |
| **Logistic Regression** | **0.34 (cost-optimal)** | 65.2% | 42.8% | **91.2%** | 58.2% | 0.836 |
| XGBoost | 0.50 (default) | 77.9% | 59.6% | 52.4% | 55.8% | 0.830 |

Logistic Regression outperforms XGBoost on this dataset — a legitimate and common outcome given the dataset's relatively small size and largely linear relationships between features (tenure, contract type) and churn.

Lowering the decision threshold to the cost-optimal value trades precision for a large recall gain, catching 91% of real churners instead of 57% — a deliberate choice justified by the assumption that losing a customer is roughly 5x more costly than a wasted retention offer.

## Known Limitations

- Precision drops meaningfully at cost-optimal thresholds — in practice, this means many "at risk" flags will be false alarms, which is only acceptable if the retention intervention itself is low-cost (e.g. an automated email, not a phone call).
- Cost assumptions (`--cost-fn`, `--cost-fp`) are illustrative; real values should come from the business (average customer lifetime value, actual campaign costs).
- No monitoring for model drift over time — performance should be re-validated periodically as customer behavior changes.

## Roadmap / Future Work

- [ ] Config-driven pipeline (read paths/params from `configs/base.yaml` instead of CLI args)
- [ ] Persistent file logging (not just console output)
- [ ] Experiment tracking (MLflow) to compare runs over time instead of overwriting `best_model.joblib`
- [ ] Input data validation/schema checks before prediction
- [ ] Lightweight API (FastAPI) to serve predictions instead of running scripts manually
- [ ] Drift monitoring for production deployment

## Testing & CI

```bash
pytest tests/ -v
ruff check src/
```

Every push runs linting and the full test suite via GitHub Actions (`.github/workflows/ci.yml`).

## License

See [LICENSE](LICENSE).
