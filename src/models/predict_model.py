import logging
import sys
from pathlib import Path

import click
import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.make_dataset import clean_data
from data.validate import validate_input_data
from monitoring import check_drift


def align_columns(df_encoded: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """Ensure new data has exactly the same columns (and order) as training data.

    Missing columns (categories not seen in this batch) are added as 0.
    Extra columns (categories not seen during training) are dropped.
    """
    df_aligned = df_encoded.reindex(columns=reference_columns, fill_value=0)
    return df_aligned


def prepare_features(df_raw: pd.DataFrame, scaler, reference_columns: list) -> pd.DataFrame:
    """Run raw new customer data through the same cleaning + encoding + scaling
    pipeline used during training."""
    df_clean = clean_data(df_raw)

    has_churn = "Churn" in df_clean.columns
    if has_churn:
        df_clean = df_clean.drop(columns=["Churn"])

    categorical_cols = df_clean.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = df_clean.select_dtypes(exclude=["object", "string"]).columns.tolist()

    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    df_aligned = align_columns(df_encoded, reference_columns)

    numeric_cols_present = [c for c in numeric_cols if c in df_aligned.columns]
    df_aligned[numeric_cols_present] = scaler.transform(df_aligned[numeric_cols_present])

    return df_aligned


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--model-path", default="models/best_model.joblib", help="Path to saved model bundle")
@click.option("--scaler-path", default="data/processed/scaler.joblib", help="Path to saved scaler")
@click.option(
    "--reference-path",
    default="data/processed/X_train.csv",
    help="Path to training features (for column alignment)",
)
def main(input_filepath, output_filepath, model_path, scaler_path, reference_path):
    """Predict churn for new customers using the saved model."""
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model bundle from {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    threshold = bundle["threshold"]
    logger.info(f"Using decision threshold: {threshold:.2f}")

    scaler = joblib.load(scaler_path)
    reference_columns = pd.read_csv(reference_path, nrows=0).columns.tolist()

    logger.info(f"Loading new data from {input_filepath}")
    df_raw = pd.read_csv(input_filepath)
    logger.info(f"Loaded {len(df_raw)} customers")

    logger.info("Validating input data...")
    validate_input_data(df_raw)
    logger.info("Input data passed validation")

    baseline_path = "models/baseline_stats.json"
    if Path(baseline_path).exists():
        logger.info("Checking for data drift against training baseline...")
        drift_results = check_drift(df_raw, baseline_path)
        n_drifted = sum(1 for r in drift_results.values() if r["drifted"])
        if n_drifted > 0:
            logger.warning(
                f"Drift detected in {n_drifted} column(s) -- consider reviewing model performance"
            )
        else:
            logger.info("No significant drift detected")
    else:
        logger.info("No baseline stats found; skipping drift check")

    X_new = prepare_features(df_raw, scaler, reference_columns)

    probs = model.predict_proba(X_new)[:, 1]
    preds = (probs >= threshold).astype(int)

    results = df_raw.copy()
    results["churn_probability"] = probs
    results["churn_prediction"] = preds
    results["churn_prediction_label"] = results["churn_prediction"].map({1: "Churn", 0: "No Churn"})

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_filepath, index=False)

    n_flagged = preds.sum()
    logger.info(
        f"Flagged {n_flagged} of {len(preds)} customers as likely to churn ({n_flagged/len(preds):.1%})"
    )
    logger.info(f"Saved predictions to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
