"""Input data validation for the churn prediction pipeline."""

import pandas as pd

EXPECTED_COLUMNS = {
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "object",  # raw data has this as string; cleaned later
}

VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "PaperlessBilling": {"Yes", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "InternetService": {"DSL", "Fiber optic", "No"},
}


class DataValidationError(Exception):
    """Raised when input data fails validation checks."""


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate a raw customer dataframe before running inference.

    Raises DataValidationError with a clear message if validation fails.
    Does not modify the dataframe.
    """
    errors = []

    # 1. Check for empty input
    if len(df) == 0:
        raise DataValidationError("Input data is empty (0 rows).")

    # 2. Check required columns are present
    missing_cols = set(EXPECTED_COLUMNS.keys()) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")

    # 3. Check for unexpected null values in critical columns
    critical_cols = ["tenure", "MonthlyCharges", "Contract"]
    for col in critical_cols:
        if col in df.columns and df[col].isnull().any():
            n_nulls = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {n_nulls} missing value(s), which is not allowed.")

    # 4. Check categorical columns only contain known values
    for col, valid_values in VALID_CATEGORIES.items():
        if col in df.columns:
            actual_values = set(df[col].dropna().unique())
            unexpected = actual_values - valid_values
            if unexpected:
                errors.append(
                    f"Column '{col}' contains unexpected values {sorted(unexpected)}; "
                    f"expected one of {sorted(valid_values)}"
                )

    # 5. Check tenure and MonthlyCharges are non-negative
    if "tenure" in df.columns and (df["tenure"] < 0).any():
        errors.append("Column 'tenure' contains negative values, which is invalid.")
    if "MonthlyCharges" in df.columns and (df["MonthlyCharges"] < 0).any():
        errors.append("Column 'MonthlyCharges' contains negative values, which is invalid.")

    if errors:
        error_message = "Input data validation failed:\n  - " + "\n  - ".join(errors)
        raise DataValidationError(error_message)
