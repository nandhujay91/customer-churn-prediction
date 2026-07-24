import logging
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def evaluate_model(name, y_test, preds, probs):
    """Return a metrics dict for given predictions and probabilities."""
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def expected_cost(y_test, preds, cost_fn, cost_fp):
    """Total business cost for a set of predictions.

    cost_fn: cost of missing a real churner (lost customer)
    cost_fp: cost of a wasted retention offer on a non-churner
    """
    _tn, fp, fn, _tp = confusion_matrix(y_test, preds).ravel()
    return fn * cost_fn + fp * cost_fp


def find_min_cost_threshold(y_test, probs, cost_fn, cost_fp):
    """Search thresholds and return the one with the lowest total business cost."""
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= threshold).astype(int)
        cost = expected_cost(y_test, preds, cost_fn, cost_fp)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold, best_cost


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--cost-fn", default=500.0, help="Cost of missing a real churner ($)")
@click.option("--cost-fp", default=50.0, help="Cost of a wasted retention offer ($)")
def main(data_dir, output_dir, cost_fn, cost_fp):
    """Train models, pick decision threshold by minimum business cost, save the best."""
    logger = logging.getLogger(__name__)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cost assumptions: false negative=${cost_fn:.0f}, false positive=${cost_fp:.0f}")

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    candidates = {}

    # --- Logistic Regression (class-weighted) ---
    logger.info("Training Logistic Regression (class-weighted)...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_reg.fit(X_train, y_train)
    lr_probs = log_reg.predict_proba(X_test)[:, 1]
    candidates["LogisticRegression"] = (log_reg, lr_probs)

    # --- XGBoost (class-weighted) ---
    logger.info("Training XGBoost (class-weighted)...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=pos_weight,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    candidates["XGBoost"] = (xgb, xgb_probs)

    # --- For each model: default threshold + cost-optimal threshold ---
    results = []
    saved_candidates = {}

    for name, (model, probs) in candidates.items():
        default_preds = (probs >= 0.5).astype(int)
        default_cost = expected_cost(y_test, default_preds, cost_fn, cost_fp)
        row = evaluate_model(f"{name} (threshold=0.50)", y_test, default_preds, probs)
        row["total_cost"] = default_cost
        results.append(row)
        saved_candidates[f"{name} (threshold=0.50)"] = (model, 0.5, default_cost)

        best_thresh, best_cost = find_min_cost_threshold(y_test, probs, cost_fn, cost_fp)
        best_preds = (probs >= best_thresh).astype(int)
        row = evaluate_model(
            f"{name} (threshold={best_thresh:.2f}, cost-optimal)", y_test, best_preds, probs
        )
        row["total_cost"] = best_cost
        results.append(row)
        saved_candidates[f"{name} (threshold={best_thresh:.2f}, cost-optimal)"] = (
            model,
            best_thresh,
            best_cost,
        )

    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n" + results_df.to_string())
    results_df.to_csv(output_dir / "model_comparison.csv")

    # --- Pick the overall lowest-cost model+threshold combo ---
    best_name = results_df["total_cost"].idxmin()
    best_model, best_thresh, best_cost = saved_candidates[best_name]

    joblib.dump(
        {"model": best_model, "threshold": best_thresh, "cost_fn": cost_fn, "cost_fp": cost_fp},
        output_dir / "best_model.joblib",
    )
    logger.info(f"Best model (lowest business cost): {best_name}")
    logger.info(f"  Total cost on test set: ${best_cost:,.0f}")
    logger.info(
        f"  Recall={results_df.loc[best_name, 'recall']:.4f}, "
        f"Precision={results_df.loc[best_name, 'precision']:.4f}"
    )
    logger.info(f"Saved to {output_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
