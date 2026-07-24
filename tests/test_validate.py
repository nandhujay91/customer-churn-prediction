import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.validate import DataValidationError, validate_input_data


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "gender": ["Female"],
            "SeniorCitizen": [0],
            "Partner": ["No"],
            "Dependents": ["No"],
            "tenure": [12],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["Yes"],
            "TechSupport": ["No"],
            "StreamingTV": ["Yes"],
            "StreamingMovies": ["No"],
            "Contract": ["One year"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Bank transfer (automatic)"],
            "MonthlyCharges": [70.5],
            "TotalCharges": ["840.0"],
        }
    )


def test_validate_passes_on_valid_data(valid_df):
    validate_input_data(valid_df)  # should not raise


def test_validate_rejects_missing_column(valid_df):
    df = valid_df.drop(columns=["Contract"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_input_data(df)


def test_validate_rejects_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError, match="empty"):
        validate_input_data(empty_df)


def test_validate_rejects_invalid_category(valid_df):
    df = valid_df.copy()
    df["Contract"] = ["Invalid Contract Type"]
    with pytest.raises(DataValidationError, match="unexpected values"):
        validate_input_data(df)


def test_validate_rejects_negative_tenure(valid_df):
    df = valid_df.copy()
    df["tenure"] = [-5]
    with pytest.raises(DataValidationError, match="negative"):
        validate_input_data(df)
