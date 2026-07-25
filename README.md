# Customer Churn Prediction

A production-style machine learning pipeline that predicts which telecom customers are likely to churn, using cost-based decision thresholds to align model behavior with real business economics.

## Overview

This project trains and compares two classifiers (Logistic Regression and XGBoost) on the IBM Telco Customer Churn dataset, then selects a decision threshold that minimizes total business cost -- not just a generic metric like accuracy or F1. The result: a model whose predictions are tuned to how expensive it actually is to lose a customer vs. wasting a retention offer on someone who was never going to leave.

## Dataset

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
7,043 customers, 19 features (contract type, tenure, monthly charges, internet service, payment method, etc.), binary churn label.

## Project Structure
customer_churn_prediction/
|-- data/
| |-- raw/ # original Kaggle CSV
| |-- processed/ # cleaned data, train/test splits, scaler
|-- models/ # trained model, baseline stats, comparison metrics
|-- src/
| |-- data/make_dataset.py # load + clean raw data
| |-- data/validate.py # input validation
| |-- features/build_features.py # encode, scale, split
| |-- models/
| | |-- train_model.py # train + cost-optimal threshold tuning + MLflow logging
| | |-- predict_model.py # inference on new customers, with validation + drift check
| |-- monitoring.py # drift detection
| |-- api.py # FastAPI serving layer
|-- tests/ # unit tests
|-- .github/workflows/ # CI: lint + test on every push
|-- Dockerfile # 3-stage containerized pipeline (Chainguard base, test-gated)
|-- configs/base.yaml # pipeline configuration
## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

## Usage

Run each pipeline stage in order (all config-driven via `configs/base.yaml` -- no arguments required):

```bash
# 1. Clean raw data
python src/data/make_dataset.py

# 2. Build features (encode, scale, split train/test)
python src/features/build_features.py

# 3. Train models and select the cost-optimal threshold
python src/models/train_model.py

# 4. Predict on new customers
python src/models/predict_model.py data/raw/new_customers.csv data/processed/predictions.csv
```

Override any config default via CLI flags, e.g. `python src/models/train_model.py --cost-fn 1000 --cost-fp 20`.

## Methodology

1. **Cleaning:** convert `TotalCharges` to numeric, drop rows with missing values, drop the customer ID, encode the target as binary (skipped gracefully if `Churn` isn't present, e.g. for new customers being scored).
2. **Feature engineering:** one-hot encode categorical variables, standardize numeric variables.
3. **Modeling:** train Logistic Regression and XGBoost, both with class weighting to address the ~27% churn / 73% no-churn imbalance.
4. **Threshold selection:** rather than using the default 0.5 cutoff, search thresholds and pick the one that **minimizes total expected cost** -- `(false negatives x cost of losing a customer) + (false positives x cost of a wasted offer)`.
5. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, and total business cost on a held-out test set.

## Results

At default cost assumptions ($500 per missed churner, $100 per wasted offer):

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 (default) | 80.4% | 64.9% | 57.0% | 60.7% | 0.836 |
| **Logistic Regression** | **0.34 (cost-optimal)** | 65.2% | 42.8% | **91.2%** | 58.2% | 0.836 |
| XGBoost | 0.50 (default) | 77.9% | 59.6% | 52.4% | 55.8% | 0.830 |

Logistic Regression outperforms XGBoost on this dataset -- a legitimate and common outcome given the dataset's relatively small size and largely linear relationships between features (tenure, contract type) and churn.

Lowering the decision threshold to the cost-optimal value trades precision for a large recall gain, catching 91% of real churners instead of 57% -- a deliberate choice justified by the assumption that losing a customer is roughly 5x more costly than a wasted retention offer.

## Known Limitations

- Precision drops meaningfully at cost-optimal thresholds -- in practice, this means many "at risk" flags will be false alarms, which is only acceptable if the retention intervention itself is low-cost (e.g. an automated email, not a phone call).
- Cost assumptions (`--cost-fn`, `--cost-fp`) are illustrative; real values should come from the business (average customer lifetime value, actual campaign costs).
- Drift detection covers three numeric columns via simple z-score comparison; it does not cover categorical feature drift or more sophisticated distributional shift (e.g. KL divergence, PSI).

## Roadmap / Future Work

All originally planned items are complete:

- [x] Config-driven pipeline (read paths/params from `configs/base.yaml` instead of CLI args)
- [x] Persistent file logging (not just console output)
- [x] Experiment tracking (MLflow) to compare runs over time instead of overwriting `best_model.joblib`
- [x] Input data validation/schema checks before prediction
- [x] Lightweight API (FastAPI) to serve predictions instead of running scripts manually
- [x] Drift monitoring for production deployment

Possible future extensions: categorical drift detection, automated retraining triggers, authentication on the API, a proper model registry.

## Experiment Tracking

Every training run is logged to MLflow (parameters, metrics, and the trained model artifact), so results are never silently overwritten and can be compared across runs.

View tracked runs in the browser:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000). Each run shows:
- Parameters: `cost_fn`, `cost_fp`, `random_state`, selected best model
- Metrics: recall, cost, and threshold for both Logistic Regression and XGBoost, at default and cost-optimal thresholds
- Artifact: the saved `best_model`, linked to the exact run that produced it

`mlflow.db` (tracking database) is excluded from git -- it's local experiment history, not part of the codebase.

## Drift Monitoring

New prediction batches are automatically checked against the training data's statistical baseline (mean and standard deviation of `tenure`, `MonthlyCharges`, `TotalCharges`). If a column's mean shifts more than 2 standard deviations from the training baseline, it's flagged as drifted -- a signal that the model may be seeing customers unlike anything it was trained on, and its predictions should be treated with caution.

The baseline is computed once after training:

```bash
python -c "import sys; sys.path.append('src'); from monitoring import compute_baseline_stats; compute_baseline_stats('data/processed/churn_cleaned.csv')"
```

`predict_model.py` checks every batch against this baseline automatically and logs warnings for any drifted columns -- no extra steps needed at inference time.

## Serving Predictions via API

A FastAPI service exposes the trained model for real-time predictions on individual customers.

Start the server:

```bash
python -m uvicorn src.api:app --reload --port 8000
```

Open the interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try it directly in the browser.

Endpoints:
- `GET /health` -- service and model-load status
- `POST /predict` -- runs the same validation and feature preparation used in training, on a single customer record, and returns a churn probability + prediction

Invalid input (e.g. an unrecognized `Contract` value) returns a `422` with a clear validation error, rather than a server crash.

## Run with Docker

The full pipeline runs in a 3-stage container (build -> test-gate -> minimal runtime), using a Chainguard minimal-CVE Python base image. Tests and lint checks run **inside the build itself** -- if they fail, the image is never produced.

Build the image:

```bash
docker build -t churn-prediction:latest .
```

Run it, mounting local folders so data, models, and logs persist back to your machine:

```bash
docker run --rm -v "${PWD}\data:/app/data" -v "${PWD}\models:/app/models" -v "${PWD}\logs:/app/logs" churn-prediction:latest
```

(On macOS/Linux, use `$(pwd)` instead of `${PWD}`.)

## Testing & CI

```bash
pytest tests/ -v
ruff check src/
```

Every push runs linting and the full test suite via GitHub Actions (`.github/workflows/ci.yml`), and again inside the Docker build's test stage.

## License

See [LICENSE](LICENSE).
