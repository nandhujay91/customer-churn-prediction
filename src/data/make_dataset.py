import logging
from pathlib import Path

import click
import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv


def load_config(config_path="configs/base.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco churn dataframe."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


@click.command()
@click.option("--config-path", default="configs/base.yaml", help="Path to config YAML")
@click.option(
    "--input-filepath", default=None, type=click.Path(exists=True), help="Override: raw data path"
)
@click.option(
    "--output-filepath", default=None, type=click.Path(), help="Override: cleaned data output path"
)
def main(config_path, input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    Defaults come from configs/base.yaml, overridable via CLI flags.
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    config = load_config(config_path)
    input_filepath = input_filepath or config["data"]["raw_path"]
    output_filepath = output_filepath or config["data"]["cleaned_path"]

    df = pd.read_csv(input_filepath)
    logger.info(f"Raw shape: {df.shape}")

    df_clean = clean_data(df)
    logger.info(f"Cleaned shape: {df_clean.shape}")

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_filepath, index=False)
    logger.info(f"Saved cleaned data to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
