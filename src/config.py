"""Central configuration: paths, thresholds, and constants.

All modules should import values from here rather than hardcoding paths or
magic numbers (CLAUDE.md Section 7). Threshold values below are transcribed
directly from CLAUDE.md Section 6 ("Business Logic — Locked Decisions") and
must not be changed without flagging it first, per Section 8.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_customer_churn.csv"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Reproducibility -----------------------------------------------------
RANDOM_STATE = 42

# --- RFM (Section 6: RFM Scoring) -----------------------------------------
RFM_QUANTILES = 5  # 1-5 scale via pd.qcut(duplicates="drop")

# --- Churn model (Section 6: Churn Model Thresholds) ----------------------
CHURN_HIGH_RISK_THRESHOLD = 0.5  # probability > 0.5 = "High churn risk"
CV_N_SPLITS = 5  # Stratified K-Fold, given class imbalance

# --- LTV (Section 6: LTV Thresholds) --------------------------------------
LTV_HIGH_PERCENTILE = 0.75  # >= 75th percentile = High LTV
LTV_MEDIUM_PERCENTILE = 0.50  # 50th-74th percentile = Medium LTV
# < 50th percentile = Low LTV (implicit; the two cutoffs above are exhaustive)

# --- Segmentation (Section 6: Segmentation) --------------------------------
KMEANS_K_MIN = 3
KMEANS_K_MAX = 8