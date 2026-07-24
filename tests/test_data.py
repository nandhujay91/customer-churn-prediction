import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.make_dataset import clean_data


@pytest.fixture
def sample_raw_df():
    """A small fake raw dataframe mimicking the Telco churn CSV structure."""
    return pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB", "0003-CCC"],
            "gender": ["Female", "Male", "Female"],
            "tenure": [5, 20, 1],
            "MonthlyCharges": [70.5, 89.9, 45.0],
            "TotalCharges": ["352.5", "1798.0", " "],  # blank string like real data
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_clean_data_drops_customer_id(sample_raw_df):
    result = clean_data(sample_raw_df)
    assert "customerID" not in result.columns


def test_clean_data_converts_total_charges_to_numeric(sample_raw_df):
    result = clean_data(sample_raw_df)
    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])


def test_clean_data_drops_rows_with_blank_total_charges(sample_raw_df):
    result = clean_data(sample_raw_df)
    # the row with " " for TotalCharges should be dropped
    assert len(result) == 2


def test_clean_data_maps_churn_to_binary(sample_raw_df):
    result = clean_data(sample_raw_df)
    assert set(result["Churn"].unique()).issubset({0, 1})
    assert result["Churn"].dtype in ("int64", "int32")


def test_clean_data_does_not_mutate_original(sample_raw_df):
    original_columns = sample_raw_df.columns.tolist()
    clean_data(sample_raw_df)
    assert sample_raw_df.columns.tolist() == original_columns
