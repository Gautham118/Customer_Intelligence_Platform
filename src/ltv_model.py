"""XGBoost LTV regressor: train, cross-validate, evaluate, predict, tier.

Depends on feature_engineering.py's `future_ltv_12m` target (CLAUDE.md
Section 9 dependency order), and optionally on churn_model.py's predicted
churn probabilities (see "Churn exclusion" below). Segmentation output
(`Cluster_ID`, `Segment_Name`) is included as a feature if present, same
convention as churn_model.py.

Churn exclusion (CLAUDE.md Section 6, LTV Target Definition -- locked):
`future_ltv_12m` is built FROM `Churn` (a churned customer's target is
truncated via the retention_multiplier -- see feature_engineering.py). That
is target *construction* and is correct/expected. It is a different thing
from `Churn` being a *feature* of the LTV regressor, which is NOT allowed,
for two independent reasons documented in CLAUDE.md:
  1. Leakage: the true label was used to help build the target, so using it
     again as a feature to predict that same target is circular.
  2. Real-world inference: a new customer you're scoring doesn't have a
     ground-truth `Churn` value yet -- only the churn classifier's
     predicted probability is available at inference time.
If churn signal is wanted as an LTV input, use
`churn_model.predict_churn_probability()` output, attached via
`add_churn_probability_feature()` below -- never the raw label.

Other exclusions, consistent with churn_model.py's reasoning:
  - `customerID` -- identifier.
  - `future_ltv_12m` / `future_ltv_12m_tier` -- the target and its
    EDA-only tiering; the real tiers this module produces come from
    `assign_ltv_tier()` applied to *predictions*, not to the synthetic
    target column.
  - `TotalCharges` -- ~= tenure x MonthlyCharges, both already present as
    separate features; same collinearity reasoning as churn_model.py.

`MonthlyCharges` and `tenure` themselves ARE kept as features (CLAUDE.md
allows this -- they're inputs to the target *formula*, not the ground-truth
label), but the rest of the feature set (RFM scores, synthetic behavioral
features, segment, churn probability) ensures the model has other drivers
to learn from rather than trivially reconstructing the formula it was built
from.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from src.config import MODELS_DIR, RANDOM_STATE
from src.feature_engineering import assign_ltv_tier

logger = logging.getLogger(__name__)

TARGET_COL = "future_ltv_12m"

CHURN_PROBABILITY_COL = "churn_probability"

# Columns that must never enter the LTV feature matrix -- see module
# docstring for the reasoning behind each.
EXCLUDED_COLUMNS = {
    "customerID",           # identifier
    "Churn",                 # ground-truth label -- see module docstring
    "future_ltv_12m",        # target itself
    "future_ltv_12m_tier",   # EDA-only tier on the synthetic target
    "TotalCharges",          # leakage risk: ~= tenure x MonthlyCharges
}

LTV_MODEL_PATH = MODELS_DIR / "ltv_model.joblib"


# --------------------------------------------------------------------------
# Optional churn-probability feature
# --------------------------------------------------------------------------

def add_churn_probability_feature(
    df: pd.DataFrame, churn_proba: np.ndarray
) -> pd.DataFrame:
    """Attach the churn classifier's predicted probability as an LTV input.

    This is the ONLY sanctioned way churn risk enters the LTV feature set
    (CLAUDE.md Section 6) -- never the ground-truth `Churn` label. Optional:
    if this is never called, the LTV model simply trains without a churn
    signal feature, which is also valid.

    Args:
        df: Feature-engineered dataframe, row order matching `churn_proba`.
        churn_proba: Predicted churn probabilities, e.g. from
            `churn_model.predict_churn_probability()`, aligned by position
            to ``df``'s rows.

    Returns:
        A copy of ``df`` with a `churn_probability` column added.

    Raises:
        ValueError: If lengths don't match.
    """
    if len(churn_proba) != len(df):
        raise ValueError(
            f"churn_proba length ({len(churn_proba)}) does not match "
            f"df length ({len(df)}) -- must be row-aligned."
        )
    df = df.copy()
    df[CHURN_PROBABILITY_COL] = churn_proba
    logger.info("Attached churn_probability feature (mean=%.3f)", df[CHURN_PROBABILITY_COL].mean())
    return df


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """List the columns eligible as LTV model features.

    Everything in ``df`` except `EXCLUDED_COLUMNS`. Inclusive of
    segmentation output and `churn_probability` when present -- see module
    docstring.

    Args:
        df: Feature-engineered dataframe (post `engineer_features`),
            optionally with `churn_probability` attached.

    Returns:
        Sorted list of feature column names.
    """
    cols = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    return sorted(cols)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the (X, y) matrix for the LTV regressor.

    String/object columns are cast to `category` dtype for XGBoost's native
    categorical support, same convention as churn_model.py.

    Args:
        df: Feature-engineered dataframe with `future_ltv_12m`.

    Returns:
        Tuple of (X, y): X is the feature dataframe with object columns cast
        to `category` dtype; y is the `future_ltv_12m` float Series.

    Raises:
        KeyError: If `future_ltv_12m` is missing from ``df``.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(
            f"'{TARGET_COL}' column not found -- call "
            "feature_engineering.add_ltv_target(df) before prepare_features()."
        )

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    y = df[TARGET_COL].astype(float)
    logger.info(
        "Prepared LTV feature matrix: %d rows, %d features (%d categorical), "
        "target mean=%.2f",
        len(X), X.shape[1], len(X.select_dtypes(include="category").columns),
        y.mean(),
    )
    return X, y


def build_model(random_state: int = RANDOM_STATE) -> XGBRegressor:
    """Construct an XGBoost regressor configured for this project.

    Args:
        random_state: Fixed seed for reproducibility (CLAUDE.md Section 7).

    Returns:
        An unfit `XGBRegressor`, configured for native categorical support.
    """
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Score predictions on the metrics CLAUDE.md Section 6 specifies.

    MAE, RMSE, and R² -- not accuracy on threshold buckets (that check is
    left to LTV-tier-level validation elsewhere, not here).

    Args:
        y_true: Ground-truth `future_ltv_12m` values.
        y_pred: Predicted LTV values.

    Returns:
        Dict with `mae`, `rmse`, `r2`, `mape` keys (floats). `mape` is
        reported since it's cheap to compute and optional per CLAUDE.md,
        but MAE/RMSE/R2 are the required three.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else float("nan")

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
    logger.info(
        "Evaluation: MAE=%.2f, RMSE=%.2f, R2=%.3f, MAPE=%.1f%%",
        mae, rmse, r2, mape,
    )
    return metrics


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def cross_validate_ltv_model(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Plain K-Fold CV for the LTV regressor.

    Unlike churn_model.py, no stratification is needed here -- there's no
    class imbalance concern for a continuous regression target. Uses the
    same `n_splits=5` convention as the churn model's CV for consistency,
    though CLAUDE.md does not lock this number for the LTV model
    specifically (my default).

    Args:
        df: Feature-engineered dataframe with `future_ltv_12m`, optionally
            with `churn_probability` attached.
        n_splits: Number of K-Fold splits.
        random_state: Fixed seed for the fold split and each fold's model.

    Returns:
        Dataframe with one row per fold (`fold`, `mae`, `rmse`, `r2`,
        `mape`).
    """
    X, y = prepare_features(df)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        metrics = evaluate_predictions(y_val, y_pred)
        metrics["fold"] = fold
        rows.append(metrics)
        logger.info("Fold %d/%d complete", fold, n_splits)

    results = pd.DataFrame(rows)[["fold", "mae", "rmse", "r2", "mape"]]
    mean_row = results[["mae", "rmse", "r2", "mape"]].mean()
    logger.info(
        "CV complete (%d folds). Mean MAE=%.2f, RMSE=%.2f, R2=%.3f, MAPE=%.1f%%",
        n_splits, mean_row["mae"], mean_row["rmse"], mean_row["r2"], mean_row["mape"],
    )
    return results


# --------------------------------------------------------------------------
# Final model (train/test split)
# --------------------------------------------------------------------------

def train_final_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[XGBRegressor, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, float]]:
    """Train the final LTV regressor on a single train/test split.

    Mirrors churn_model.train_final_model()'s shape for consistency, so
    explainability.py can consume both the same way.

    Args:
        df: Feature-engineered dataframe with `future_ltv_12m`, optionally
            with `churn_probability` attached.
        test_size: Fraction of rows held out for testing.
        random_state: Fixed seed for the split and the model.

    Returns:
        Tuple of `(model, X_train, X_test, y_train, y_test, test_metrics)`.
    """
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, test_metrics


# --------------------------------------------------------------------------
# Prediction + tiering
# --------------------------------------------------------------------------

def predict_ltv(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict LTV values for a feature matrix.

    Args:
        model: A fitted `XGBRegressor` (e.g. from `train_final_model`).
        X: Feature matrix built the same way as training (see
            `prepare_features` -- object columns must be `category` dtype).

    Returns:
        Array of predicted `future_ltv_12m` values.
    """
    return model.predict(X)


def tier_predictions(y_pred: np.ndarray, index: Optional[pd.Index] = None) -> pd.Series:
    """Bucket predicted LTV values into the three locked tiers.

    Reuses `feature_engineering.assign_ltv_tier()` rather than
    reimplementing the percentile logic (CLAUDE.md Section 6: High >= 75th
    percentile, Medium 50th-74th, Low < 50th) -- applied here to the
    regressor's actual *predictions*, which is what CLAUDE.md means by
    "predicted LTV values in the dataset," as opposed to the synthetic
    target's EDA-only tiering done earlier in feature_engineering.py.

    Args:
        y_pred: Predicted LTV values.
        index: Optional index to attach to the returned Series (e.g. the
            corresponding `customerID` or original dataframe index).

    Returns:
        String Series with values in {"High", "Medium", "Low"}.
    """
    values = pd.Series(y_pred, index=index)
    return assign_ltv_tier(values)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_model(model: XGBRegressor, path: Optional[Path] = None) -> Path:
    """Save a fitted model to disk via joblib.

    Args:
        model: Fitted `XGBRegressor`.
        path: Optional override path. Defaults to `LTV_MODEL_PATH`
            (`models/ltv_model.joblib`).

    Returns:
        The path the model was saved to.
    """
    save_path = path or LTV_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("Saved LTV model to %s", save_path)
    return save_path


def load_model(path: Optional[Path] = None) -> XGBRegressor:
    """Load a previously saved LTV model from disk.

    Args:
        path: Optional override path. Defaults to `LTV_MODEL_PATH`.

    Returns:
        The loaded `XGBRegressor`.

    Raises:
        FileNotFoundError: If no saved model exists at the path.
    """
    load_path = path or LTV_MODEL_PATH
    if not load_path.exists():
        raise FileNotFoundError(
            f"No saved LTV model at {load_path} -- run train_final_model() "
            "and save_model() first."
        )
    logger.info("Loading LTV model from %s", load_path)
    return joblib.load(load_path)