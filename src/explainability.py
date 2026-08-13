"""SHAP explainability: global feature importance + per-customer narratives.

Works generically against either the churn classifier (churn_model.py) or
the LTV regressor (ltv_model.py) -- both are XGBoost tree models, so a
single `shap.TreeExplainer` codepath covers both (CLAUDE.md Section 3, item
5: "SHAP (global + local/waterfall explanations)").

Depends on `train_final_model()` from churn_model.py / ltv_model.py per
CLAUDE.md Section 9's dependency order: this module consumes the
`(model, X_test, y_test)` those functions return, so SHAP explanations are
built against genuinely held-out rows, not training rows the model has
already seen.

Section 9's Definition-of-Done requirement -- "SHAP explanations generated
for at least 3 example customers, with human-readable narrative text, not
just raw plots" -- is what `generate_customer_narrative()` /
`explain_customers()` exist for; the plotting functions alone would not
satisfy that on their own.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

from src.config import REPORTS_FIGURES_DIR

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Explainer construction
# --------------------------------------------------------------------------

def get_shap_explainer(model) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for a fitted XGBoost model.

    Works for both `XGBClassifier` (churn_model.py) and `XGBRegressor`
    (ltv_model.py) -- TreeExplainer dispatches on the underlying booster,
    not the sklearn wrapper class.

    Args:
        model: A fitted `XGBClassifier` or `XGBRegressor`.

    Returns:
        A `shap.TreeExplainer` bound to ``model``.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for a feature matrix.

    Args:
        explainer: Output of `get_shap_explainer`.
        X: Feature matrix, built the same way as training (`category`
            dtype for object columns -- see `churn_model.prepare_features`
            / `ltv_model.prepare_features`).

    Returns:
        Array of shape (n_rows, n_features). For a binary classifier this
        is already the "probability of the positive class" contribution
        space when `explainer` was built on an `XGBClassifier` with default
        settings (SHAP handles the margin-vs-probability space internally
        for TreeExplainer + XGBoost).
    """
    shap_values = explainer.shap_values(X)
    # Some SHAP/XGBoost version combinations return a list (one array per
    # class) for binary classifiers instead of a single 2D array -- normalize
    # to the positive-class array so downstream code has one consistent shape.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    logger.info("Computed SHAP values: shape=%s", shap_values.shape)
    return shap_values


# --------------------------------------------------------------------------
# Global explanations
# --------------------------------------------------------------------------

def global_feature_importance(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value across all rows.

    This is the "global" half of Section 3's SHAP requirement -- one number
    per feature summarizing its typical impact magnitude across the whole
    test set, independent of direction.

    Args:
        shap_values: Output of `compute_shap_values`.
        X: The same feature matrix `shap_values` was computed on (for
            column names).

    Returns:
        Dataframe with `feature` and `mean_abs_shap`, sorted descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance


def plot_global_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_path: Optional[Path] = None,
    max_display: int = 15,
) -> Path:
    """Save a SHAP beeswarm summary plot to `reports/figures/`.

    Args:
        shap_values: Output of `compute_shap_values`.
        X: The same feature matrix `shap_values` was computed on.
        output_path: Optional override path. Defaults to
            `reports/figures/shap_global_summary.png`.
        max_display: Max number of features shown on the plot.

    Returns:
        The path the figure was saved to.
    """
    import matplotlib.pyplot as plt

    save_path = output_path or (REPORTS_FIGURES_DIR / "shap_global_summary.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP global summary plot to %s", save_path)
    return save_path


# --------------------------------------------------------------------------
# Per-customer (local) explanations
# --------------------------------------------------------------------------

def plot_waterfall_for_customer(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    row_position: int,
    output_path: Optional[Path] = None,
) -> Path:
    """Save a SHAP waterfall plot for one customer.

    Args:
        explainer: Output of `get_shap_explainer` (used for `expected_value`).
        shap_values: Output of `compute_shap_values`.
        X: The same feature matrix `shap_values` was computed on.
        row_position: Positional (0-indexed) row in ``X`` to explain -- NOT
            a `customerID`; use `X.index.get_loc(...)` if starting from a
            label.
        output_path: Optional override path. Defaults to
            `reports/figures/shap_waterfall_{row_position}.png`.

    Returns:
        The path the figure was saved to.
    """
    import matplotlib.pyplot as plt

    save_path = output_path or (
        REPORTS_FIGURES_DIR / f"shap_waterfall_{row_position}.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1]

    explanation = shap.Explanation(
        values=shap_values[row_position],
        base_values=expected_value,
        data=X.iloc[row_position],
        feature_names=X.columns.tolist(),
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP waterfall plot for row %d to %s", row_position, save_path)
    return save_path


def generate_customer_narrative(
    X_row: pd.Series,
    shap_row: np.ndarray,
    base_value: float,
    prediction: float,
    customer_id: Optional[str] = None,
    is_classifier: bool = True,
    top_n: int = 4,
) -> str:
    """Turn one customer's SHAP values into a human-readable paragraph.

    This is the piece Section 9 explicitly requires beyond raw plots --
    an interviewer-facing sentence explaining *why* the model landed on
    this prediction for this specific customer, not just a chart.

    Args:
        X_row: The customer's feature values (one row of the feature
            matrix SHAP was computed on).
        shap_row: That customer's SHAP values (one row of `shap_values`).
        base_value: The explainer's expected value (model's average
            output over the training/background data).
        prediction: The model's actual prediction for this customer
            (probability for churn, dollar amount for LTV).
        customer_id: Optional label for the narrative (e.g. `customerID`).
        is_classifier: If True, phrases the narrative in churn-probability
            terms; if False, in dollar-LTV terms.
        top_n: How many top contributing features to name.

    Returns:
        A short narrative string.
    """
    order = np.argsort(-np.abs(shap_row))[:top_n]
    label = customer_id if customer_id is not None else "This customer"

    if is_classifier:
        pred_str = f"a predicted churn probability of {prediction:.0%}"
    else:
        pred_str = f"a predicted 12-month LTV of ${prediction:,.0f}"

    parts = [f"{label} has {pred_str} (baseline: {base_value:.2f})."]

    pushes_up, pushes_down = [], []
    for i in order:
        feat = X_row.index[i]
        val = X_row.iloc[i]
        if isinstance(val, (float, np.floating)):
            val = round(float(val), 2)
        contribution = shap_row[i]
        direction = pushes_up if contribution > 0 else pushes_down
        direction.append(f"{feat}={val}")

    if pushes_up:
        verb = "increasing churn risk" if is_classifier else "increasing predicted LTV"
        parts.append(f"Factors {verb}: " + ", ".join(pushes_up) + ".")
    if pushes_down:
        verb = "reducing churn risk" if is_classifier else "reducing predicted LTV"
        parts.append(f"Factors {verb}: " + ", ".join(pushes_down) + ".")

    return " ".join(parts)


def explain_customers(
    model,
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    customer_ids: Optional[pd.Series] = None,
    n: int = 3,
    is_classifier: bool = True,
    random_state: int = 42,
) -> list[dict]:
    """Generate full local explanations (narrative + SHAP row) for n customers.

    Convenience orchestrator satisfying Section 9's "at least 3 example
    customers" requirement in one call -- samples `n` rows from ``X``,
    computes their predictions and SHAP contributions, and returns a
    narrative for each.

    Args:
        model: A fitted `XGBClassifier` or `XGBRegressor`.
        explainer: Output of `get_shap_explainer` (same model).
        X: Feature matrix (typically `X_test` from `train_final_model`).
        customer_ids: Optional Series of IDs aligned to ``X``'s rows, used
            to label each narrative. If None, positional labels are used.
        n: Number of customers to sample and explain.
        is_classifier: Passed through to `generate_customer_narrative`.
        random_state: Seed for the sample selection.

    Returns:
        List of dicts, one per sampled customer, each with `customer_id`,
        `row_position`, `prediction`, `narrative`, and `shap_values`.
    """
    rng = np.random.default_rng(random_state)
    n = min(n, len(X))
    positions = rng.choice(len(X), size=n, replace=False)

    shap_values = compute_shap_values(explainer, X)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1]

    if is_classifier:
        predictions = model.predict_proba(X)[:, 1]
    else:
        predictions = model.predict(X)

    results = []
    for pos in positions:
        pos = int(pos)
        cid = customer_ids.iloc[pos] if customer_ids is not None else f"row_{pos}"
        narrative = generate_customer_narrative(
            X_row=X.iloc[pos],
            shap_row=shap_values[pos],
            base_value=float(expected_value),
            prediction=float(predictions[pos]),
            customer_id=cid,
            is_classifier=is_classifier,
        )
        results.append({
            "customer_id": cid,
            "row_position": pos,
            "prediction": float(predictions[pos]),
            "narrative": narrative,
            "shap_values": shap_values[pos],
        })
        logger.info("Explained customer %s: %s", cid, narrative)

    return results