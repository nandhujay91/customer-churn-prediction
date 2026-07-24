import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from features.build_features import build_features


@pytest.fixture
def sample_clean_df():
    """A small fake cleaned dataframe (post clean_data output)."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male"],
            "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
            "tenure": [5, 20, 15, 1],
            "MonthlyCharges": [70.5, 89.9, 60.0, 45.0],
            "TotalCharges": [352.5, 1798.0, 900.0, 45.0],
            "Churn": [1, 0, 0, 1],
        }
    )


def test_build_features_separates_target(sample_clean_df):
    X, y, scaler = build_features(sample_clean_df)
    assert "Churn" not in X.columns
    assert list(y) == [1, 0, 0, 1]


def test_build_features_one_hot_encodes_categoricals(sample_clean_df):
    X, y, scaler = build_features(sample_clean_df)
    # original categorical columns should be gone, replaced by dummy columns
    assert "gender" not in X.columns
    assert "Contract" not in X.columns
    assert any(col.startswith("gender_") for col in X.columns)


def test_build_features_no_missing_values(sample_clean_df):
    X, y, scaler = build_features(sample_clean_df)
    assert X.isnull().sum().sum() == 0


def test_build_features_scales_numeric_columns(sample_clean_df):
    X, y, scaler = build_features(sample_clean_df)
    # scaled tenure should have mean approx 0
    assert abs(X["tenure"].mean()) < 1e-6


def test_build_features_output_row_count_matches_input(sample_clean_df):
    X, y, scaler = build_features(sample_clean_df)
    assert len(X) == len(sample_clean_df)
    assert len(y) == len(sample_clean_df)
