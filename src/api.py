"""FastAPI serving layer for the churn prediction model."""

import logging
import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent))

from data.validate import DataValidationError, validate_input_data
from models.predict_model import prepare_features

logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)

MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "data/processed/scaler.joblib"
REFERENCE_PATH = "data/processed/X_train.csv"

_model_bundle = None
_scaler = None
_reference_columns = None


def load_artifacts():
    """Load model, scaler, and reference columns once at startup."""
    global _model_bundle, _scaler, _reference_columns
    logger.info(f"Loading model bundle from {MODEL_PATH}")
    _model_bundle = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _reference_columns = pd.read_csv(REFERENCE_PATH, nrows=0).columns.tolist()
    logger.info(f"Model loaded. Decision threshold: {_model_bundle['threshold']:.2f}")


@app.on_event("startup")
def startup_event():
    load_artifacts()


class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": "1020.5",
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    churn_prediction_label: str
    decision_threshold: float


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _model_bundle is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Predict churn probability for a single customer."""
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    df_raw = pd.DataFrame([customer.model_dump()])

    try:
        validate_input_data(df_raw)
    except DataValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    X_new = prepare_features(df_raw, _scaler, _reference_columns)

    model = _model_bundle["model"]
    threshold = _model_bundle["threshold"]

    prob = float(model.predict_proba(X_new)[:, 1][0])
    pred = int(prob >= threshold)
    label = "Churn" if pred == 1 else "No Churn"

    return PredictionResponse(
        churn_probability=prob,
        churn_prediction=pred,
        churn_prediction_label=label,
        decision_threshold=threshold,
    )
