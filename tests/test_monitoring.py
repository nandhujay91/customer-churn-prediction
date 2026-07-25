import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from monitoring import check_drift, compute_baseline_stats


@pytest.fixture
def baseline_file(tmp_path):
    """Create a temporary baseline stats file for testing."""
    stats = {
        "tenure": {"mean": 30.0, "std": 20.0, "min": 1.0, "max": 72.0},
        "MonthlyCharges": {"mean": 65.0, "std": 30.0, "min": 18.0, "max": 120.0},
    }
    path = tmp_path / "baseline_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f)
    return str(path)


def test_compute_baseline_stats_creates_file(tmp_path):
    df = pd.DataFrame({"tenure": [10, 20, 30], "MonthlyCharges": [50.0, 60.0, 70.0]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    output_path = tmp_path / "baseline.json"

    stats = compute_baseline_stats(str(csv_path), str(output_path))

    assert output_path.exists()
    assert "tenure" in stats
    assert stats["tenure"]["mean"] == 20.0


def test_check_drift_detects_no_drift_on_similar_data(baseline_file):
    df = pd.DataFrame({"tenure": [31, 29, 30], "MonthlyCharges": [64.0, 66.0, 65.0]})
    results = check_drift(df, baseline_file)
    assert results["tenure"]["drifted"] is False
    assert results["MonthlyCharges"]["drifted"] is False


def test_check_drift_detects_real_drift(baseline_file):
    df = pd.DataFrame({"tenure": [200, 210, 205], "MonthlyCharges": [65.0, 64.0, 66.0]})
    results = check_drift(df, baseline_file)
    assert results["tenure"]["drifted"] is True
    assert results["MonthlyCharges"]["drifted"] is False


def test_check_drift_ignores_columns_not_in_new_data(baseline_file):
    df = pd.DataFrame({"tenure": [30, 31, 29]})
    results = check_drift(df, baseline_file)
    assert "tenure" in results
    assert "MonthlyCharges" not in results
