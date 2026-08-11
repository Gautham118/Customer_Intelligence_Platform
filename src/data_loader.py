"""Load the raw Telco Customer Churn CSV and validate its schema.
the raw file lives at data/raw/telco_customer_churn.csv
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw Telco churn CSV from disk.

    Args:
        path: Optional override for the CSV path. Defaults to
            ``config.RAW_DATA_PATH``.

    Returns:
        The raw dataframe, unmodified (no cleaning is applied here --
        see preprocessing.py for that).

    Raises:
        FileNotFoundError: If the CSV does not exist at the expected path.
            Per CLAUDE.md, do not fetch a replacement -- ask the user to
            place the file instead.
        ValueError: If the file exists but fails schema validation.
    """
    csv_path = path or RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw Telco CSV not found at {csv_path}. Do not download a "
            "replacement -- place the original file there and retry."
        )
    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path)
    validate_schema(df)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Run basic schema checks on the raw dataframe.

    Checks that all expected columns are present and that ``customerID``
    is unique, since every downstream module assumes one row per customer.

    Args:
        df: The raw dataframe to validate.

    Raises:
        ValueError: If expected columns are missing or ``customerID`` has
            duplicates.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    n_dupes = int(df["customerID"].duplicated().sum())
    if n_dupes:
        raise ValueError(f"Found {n_dupes} duplicate customerID values")

    logger.info(
        "Schema validation passed: %d expected columns present, "
        "customerID unique",
        len(EXPECTED_COLUMNS),
    )