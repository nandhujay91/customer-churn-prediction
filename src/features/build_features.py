import logging
from pathlib import Path

import click
import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_config(config_path="configs/base.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_features(df: pd.DataFrame):
    """One-hot encode categoricals and scale numeric columns."""
    df = df.copy()

    target = df.pop("Churn")

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = df.select_dtypes(exclude=["object", "string"]).columns.tolist()

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    return df_encoded, target, scaler


@click.command()
@click.option("--config-path", default="configs/base.yaml", help="Path to config YAML")
@click.option(
    "--input-filepath",
    default=None,
    type=click.Path(exists=True),
    help="Override: cleaned data path",
)
@click.option("--output-dir", default=None, type=click.Path(), help="Override: output directory")
def main(config_path, input_filepath, output_dir):
    """Build model-ready features from cleaned churn data and split train/test.
    Defaults come from configs/base.yaml, overridable via CLI flags.
    """
    logger = logging.getLogger(__name__)
    logger.info("building features from cleaned data")

    config = load_config(config_path)
    input_filepath = input_filepath or config["data"]["cleaned_path"]
    output_dir = Path(output_dir or config["data"]["processed_dir"])
    random_state = config.get("random_state", 42)

    df = pd.read_csv(input_filepath)
    logger.info(f"Loaded cleaned data: {df.shape}")

    X, y, scaler = build_features(df)
    logger.info(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    logger.info(f"Saved train/test splits and scaler to {output_dir}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
