import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.make_dataset import clean_data


def test_clean_data_works_without_churn_column():
    """Regression test: new customers being scored have no Churn column yet.
    clean_data() must not crash when Churn is absent."""
    df = pd.DataFrame(
        {
            "customerID": ["9999-XXX"],
            "gender": ["Female"],
            "tenure": [12],
            "MonthlyCharges": [70.5],
            "TotalCharges": ["840.0"],
        }
    )
    result = clean_data(df)
    assert "Churn" not in result.columns
    assert "customerID" not in result.columns
    assert len(result) == 1
