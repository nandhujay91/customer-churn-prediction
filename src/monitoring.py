"""Simple statistical drift detection: compares new data against the training baseline."""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("drift")

NUMERIC_COLS_TO_MONITOR = ["tenure", "MonthlyCharges", "TotalCharges"]


def compute_baseline_stats(reference_csv: str, output_path: str = "models/baseline_stats.json"):
    """Compute and save summary statistics from the training data.
    Run this once after training, to establish what "normal" looks like."""
    df = pd.read_csv(reference_csv)

    stats = {}
    for col in NUMERIC_COLS_TO_MONITOR:
        if col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved baseline drift stats to {output_path}")
    return stats


def check_drift(
    new_df: pd.DataFrame,
    baseline_path: str = "models/baseline_stats.json",
    z_threshold: float = 2.0,
):
    """Compare new data's mean for monitored columns against the training baseline.

    Flags a column as drifted if its new mean is more than `z_threshold` standard
    deviations away from the training mean (using the training std as the reference).

    Returns a dict: {column: {"drifted": bool, "baseline_mean": ..., "new_mean": ..., "z_score": ...}}
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    results = {}
    for col, base_stats in baseline.items():
        if col not in new_df.columns:
            continue

        new_mean = float(new_df[col].mean())
        std = base_stats["std"] if base_stats["std"] > 0 else 1e-9
        z_score = abs(new_mean - base_stats["mean"]) / std
        drifted = z_score > z_threshold

        results[col] = {
            "drifted": drifted,
            "baseline_mean": base_stats["mean"],
            "new_mean": new_mean,
            "z_score": round(z_score, 3),
        }

        if drifted:
            logger.warning(
                f"Drift detected in '{col}': baseline mean={base_stats['mean']:.2f}, "
                f"new mean={new_mean:.2f} (z-score={z_score:.2f})"
            )
        else:
            logger.info(f"No drift in '{col}': z-score={z_score:.2f}")

    return results
