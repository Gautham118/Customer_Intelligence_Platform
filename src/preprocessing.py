"""Cleaning and dtype fixes for the raw Telco dataframe.

Handles the one documented data quirk in CLAUDE.md Section 2: 11 rows have
a blank/whitespace TotalCharges value, all at tenure == 0 (brand-new
customers with no billing history yet).
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges to numeric and impute the known blank rows.

    TotalCharges loads as a string column because 11 rows contain a blank/
    whitespace value instead of a number. All 11 are new customers
    (tenure == 0) who haven't been billed yet, so a blank there isn't a real
    gap -- it means zero billed to date. Those rows are imputed to 0.0
    rather than dropped, since dropping would discard otherwise-complete
    customer records over a single explainable field.

    Args:
        df: Dataframe containing a raw ``TotalCharges`` column (string dtype).

    Returns:
        A copy of ``df`` with ``TotalCharges`` as float64 and the known
        blank rows imputed to 0.0.
    """
    df = df.copy()
    coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = coerced.isna()
    n_blank = int(blank_mask.sum())

    if n_blank:
        non_new_customer_blanks = int((df.loc[blank_mask, "tenure"] != 0).sum())
        if non_new_customer_blanks:
            logger.warning(
                "%d blank TotalCharges rows have tenure != 0 -- this differs "
                "from the documented quirk (blanks expected only at "
                "tenure == 0). Investigate before trusting the imputation.",
                non_new_customer_blanks,
            )
        logger.info(
            "Imputing %d blank TotalCharges rows (tenure == 0) to 0.0", n_blank
        )
        coerced = coerced.fillna(0.0)

    df["TotalCharges"] = coerced
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps on the raw Telco dataframe.

    Args:
        df: Raw dataframe as returned by ``data_loader.load_raw_data``.

    Returns:
        A cleaned dataframe ready for feature engineering: ``TotalCharges``
        is numeric with no missing values, and no rows have been dropped.
    """
    n_before = len(df)
    df = fix_total_charges(df)
    assert len(df) == n_before, "clean_data must not drop rows"
    logger.info("Cleaning complete: %d rows retained", len(df))
    return df