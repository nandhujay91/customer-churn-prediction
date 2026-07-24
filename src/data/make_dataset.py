import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco churn dataframe."""
    df = df.copy()

    # TotalCharges has blank strings for new customers -> convert & drop bad rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # customerID is just an identifier, not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Standardize target: Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

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
