"""XGBoost churn classifier: train, cross-validate, evaluate, predict.

Depends on feature_engineering.py (and, if run first, segmentation.py) per
CLAUDE.md Section 9's module dependency order: this file expects a dataframe
that has already been through `preprocessing.clean_data` and
`feature_engineering.engineer_features` (RFM scores, synthetic behavioral
features, `future_ltv_12m`). Segmentation output (`Cluster_ID`,
`Segment_Name`) is optional -- included as a feature if present, but this
module does not require it.

Leakage guardrails (CLAUDE.md Section 2 & 6, confirmed in this project's
working notes):
  - `customerID` excluded -- identifier, not signal.
  - `Churn` excluded -- it's the prediction target itself.
  - `future_ltv_12m` (and its tier) excluded -- that's the LTV regression's
    target, not a churn feature; including it here would be modeling the
    wrong direction of the causal story (LTV realization depends on churn,
    not the reverse).
  - `TotalCharges` excluded -- ~= tenure x MonthlyCharges, both of which are
    already features; leaving it in adds a near-collinear, leak-flavored
    column for no modeling benefit.

Class imbalance (CLAUDE.md Section 6): `scale_pos_weight` is the one
documented imbalance-handling technique for v1. It is recomputed from
whichever training slice is currently in view (a single train split, or one
CV fold's training rows) -- never from the full dataset before splitting --
so no information about a held-out fold's or the test set's class balance
leaks into training.

Calibration caveat (CLAUDE.md Section 6): `scale_pos_weight` improves recall
on the minority class but skews predicted probabilities away from true
calibrated probabilities. Outputs of this module are for ranking/
prioritization (who is relatively higher risk) at the documented 0.5
threshold, not literal calibrated probabilities. `CalibratedClassifierCV` is
out of scope for v1.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.config import CHURN_HIGH_RISK_THRESHOLD, CV_N_SPLITS, MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

TARGET_COL = "Churn"

# Columns that must never enter the churn feature matrix -- see module
# docstring for the reasoning behind each.
EXCLUDED_COLUMNS = {
    "customerID",       # identifier
    "Churn",             # target
    "future_ltv_12m",    # wrong-model target (LTV regressor's job)
    "future_ltv_12m_tier",  # derived from the LTV target -- same exclusion reason
    "TotalCharges",      # leakage risk: ~= tenure x MonthlyCharges
}

CHURN_MODEL_PATH = MODELS_DIR / "churn_model.joblib"


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """List the columns eligible as churn model features.

    Everything in ``df`` except `EXCLUDED_COLUMNS`. This is intentionally
    inclusive of segmentation output (`Cluster_ID`, `Segment_Name`) and the
    RFM columns when present, since those are legitimate business-derived
    signals, not leakage -- but the function works fine on a dataframe that
    hasn't been through segmentation yet.

    Args:
        df: Feature-engineered dataframe (post `engineer_features`).

    Returns:
        Sorted list of feature column names.
    """
    cols = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    return sorted(cols)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the (X, y) matrix for the churn classifier.

    String/object columns are cast to pandas `category` dtype so XGBoost's
    native categorical support (`enable_categorical=True`) can split on them
    directly, rather than manually one-hot-encoding -- avoids column
    explosion from high-cardinality string columns and keeps SHAP output
    (Section 9's explainability requirement) mapped to interpretable,
    human-readable original feature names downstream.

    Args:
        df: Feature-engineered dataframe with a `Churn` column ("Yes"/"No").

    Returns:
        Tuple of (X, y): X is the feature dataframe with object columns cast
        to `category` dtype; y is a 0/1 integer Series (1 = churned).

    Raises:
        KeyError: If `Churn` is missing from ``df``.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(
            f"'{TARGET_COL}' column not found -- prepare_features expects the "
            "raw ground-truth label to build y."
        )

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    y = (df[TARGET_COL] == "Yes").astype(int)
    logger.info(
        "Prepared feature matrix: %d rows, %d features (%d categorical), "
        "positive class rate=%.3f",
        len(X), X.shape[1], len(X.select_dtypes(include="category").columns),
        y.mean(),
    )
    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute `scale_pos_weight` (negative:positive ratio) from a label slice.

    Must be called on a *training* slice only (a train split, or one CV
    fold's training rows) -- never on the full dataset before splitting, or
    the held-out data's class balance leaks into the model's imbalance
    handling (CLAUDE.md Section 6: "set from the training data, not
    hardcoded").

    Args:
        y: 0/1 label series for the training rows only.

    Returns:
        Ratio of negative-class count to positive-class count.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("scale_pos_weight is undefined: zero positive-class rows.")
    return n_neg / n_pos


def build_model(scale_pos_weight: float, random_state: int = RANDOM_STATE) -> XGBClassifier:
    """Construct an XGBoost classifier configured for this project.

    Args:
        scale_pos_weight: Negative:positive class ratio for this training
            slice (see `compute_scale_pos_weight`).
        random_state: Fixed seed for reproducibility (CLAUDE.md Section 7).

    Returns:
        An unfit `XGBClassifier`, configured for native categorical support
        and imbalance handling via `scale_pos_weight`.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = CHURN_HIGH_RISK_THRESHOLD,
) -> dict[str, float]:
    """Score predictions on the metrics CLAUDE.md Section 6 prioritizes.

    Recall, ROC-AUC, and PR-AUC are reported -- explicitly not raw accuracy,
    since a missed churner (false negative) costs more than a wasted
    retention offer (false positive), and the ~26% positive class rate makes
    plain accuracy an uninformative, and ROC-AUC alone an optimistic-looking,
    metric (PR-AUC is the more honest one here).

    Args:
        y_true: Ground-truth 0/1 labels.
        y_proba: Predicted churn probabilities (uncalibrated -- see module
            docstring).
        threshold: Probability cutoff for the "high churn risk" class label,
            used only for the recall/confusion-matrix calculation below.
            Defaults to `config.CHURN_HIGH_RISK_THRESHOLD` (0.5).

    Returns:
        Dict with `recall`, `roc_auc`, `pr_auc` keys (floats).
    """
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
    logger.info(
        "Evaluation @ threshold=%.2f: recall=%.3f, roc_auc=%.3f, pr_auc=%.3f",
        threshold, metrics["recall"], metrics["roc_auc"], metrics["pr_auc"],
    )
    return metrics


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def cross_validate_churn_model(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stratified K-fold CV, with `scale_pos_weight` recomputed per fold.

    Plain K-Fold risks folds with very few churners given the ~26% positive
    rate (CLAUDE.md Section 6), hence Stratified K-Fold. `scale_pos_weight`
    is computed fresh from each fold's training rows only (never from the
    full dataset) so no validation-fold class balance leaks into training.

    Args:
        df: Feature-engineered dataframe with `Churn`.
        n_splits: Number of stratified folds. Defaults to
            `config.CV_N_SPLITS` (5).
        random_state: Fixed seed for the fold split and each fold's model.

    Returns:
        Dataframe with one row per fold (`fold`, `recall`, `roc_auc`,
        `pr_auc`, `scale_pos_weight`) plus a final `mean` row.
    """
    X, y = prepare_features(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        spw = compute_scale_pos_weight(y_train)
        model = build_model(scale_pos_weight=spw, random_state=random_state)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, y_proba)
        metrics.update({"fold": fold, "scale_pos_weight": spw})
        rows.append(metrics)
        logger.info("Fold %d/%d complete (scale_pos_weight=%.3f)", fold, n_splits, spw)

    results = pd.DataFrame(rows)[["fold", "scale_pos_weight", "recall", "roc_auc", "pr_auc"]]
    mean_row = results[["recall", "roc_auc", "pr_auc"]].mean()
    logger.info(
        "CV complete (%d folds). Mean recall=%.3f, roc_auc=%.3f, pr_auc=%.3f",
        n_splits, mean_row["recall"], mean_row["roc_auc"], mean_row["pr_auc"],
    )
    return results


# --------------------------------------------------------------------------
# Final model (train/test split, for downstream use by explainability.py etc.)
# --------------------------------------------------------------------------

def train_final_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[XGBClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, float]]:
    """Train the final churn model on a single train/test split.

    CV above is for evaluation only, per standard practice -- this function
    fits the model that actually gets saved and used downstream.
    `explainability.py` (per this project's module dependency order) expects
    `X_test`/`y_test` from this function to build SHAP explanations against
    genuinely held-out rows.

    `scale_pos_weight` is computed from `y_train` only, consistent with the
    CV folds above -- the test set's class balance never touches training.

    Args:
        df: Feature-engineered dataframe with `Churn`.
        test_size: Fraction of rows held out for testing.
        random_state: Fixed seed for the split and the model.

    Returns:
        Tuple of `(model, X_train, X_test, y_train, y_test, test_metrics)`.
    """
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    spw = compute_scale_pos_weight(y_train)
    model = build_model(scale_pos_weight=spw, random_state=random_state)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, y_proba)

    y_pred = (y_proba >= CHURN_HIGH_RISK_THRESHOLD).astype(int)
    logger.info(
        "Final model test-set confusion matrix:\n%s",
        confusion_matrix(y_test, y_pred),
    )
    logger.info(
        "Final model test-set classification report:\n%s",
        classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
    )

    return model, X_train, X_test, y_train, y_test, test_metrics


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict_churn_probability(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict churn probabilities for a feature matrix.

    Args:
        model: A fitted `XGBClassifier` (e.g. from `train_final_model`).
        X: Feature matrix built the same way as training (see
            `prepare_features` -- object columns must be `category` dtype).

    Returns:
        Array of predicted churn probabilities (uncalibrated -- see module
        docstring for the ranking-not-literal-probability caveat).
    """
    return model.predict_proba(X)[:, 1]


def flag_high_risk(
    y_proba: np.ndarray, threshold: float = CHURN_HIGH_RISK_THRESHOLD
) -> np.ndarray:
    """Threshold predicted probabilities into a High/Low churn risk flag.

    Args:
        y_proba: Predicted churn probabilities.
        threshold: Cutoff for "High churn risk". Defaults to
            `config.CHURN_HIGH_RISK_THRESHOLD` (0.5), documented as
            adjustable based on business cost tradeoffs (CLAUDE.md Section 6).

    Returns:
        Boolean array, True where predicted risk is "High".
    """
    return y_proba >= threshold


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_model(model: XGBClassifier, path: Optional[Path] = None) -> Path:
    """Save a fitted model to disk via joblib.

    Args:
        model: Fitted `XGBClassifier`.
        path: Optional override path. Defaults to `CHURN_MODEL_PATH`
            (`models/churn_model.joblib`).

    Returns:
        The path the model was saved to.
    """
    save_path = path or CHURN_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("Saved churn model to %s", save_path)
    return save_path


def load_model(path: Optional[Path] = None) -> XGBClassifier:
    """Load a previously saved churn model from disk.

    Args:
        path: Optional override path. Defaults to `CHURN_MODEL_PATH`.

    Returns:
        The loaded `XGBClassifier`.

    Raises:
        FileNotFoundError: If no saved model exists at the path.
    """
    load_path = path or CHURN_MODEL_PATH
    if not load_path.exists():
        raise FileNotFoundError(
            f"No saved churn model at {load_path} -- run train_final_model() "
            "and save_model() first."
        )
    logger.info("Loading churn model from %s", load_path)
    return joblib.load(load_path)