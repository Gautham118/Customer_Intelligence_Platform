"""Churn Risk x LTV priority matrix -- the canonical per-customer recommendation logic.

CLAUDE.md Section 6 resolves an apparent conflict between two tables it
defines: the Churn Risk x LTV Priority Matrix, and the Segment -> Action
table. Only the FIRST is implemented as executable logic here -- it's what
drives the per-customer recommendation shown in the dashboard's
Recommendations page (Section 9's "real, interactive table/view"
requirement). The Segment -> Action table is a narrative/strategic overlay
for segment-level dashboard messaging and the README's business narrative
only; it is intentionally NOT re-implemented as a second, potentially
contradictory piece of logic here (CLAUDE.md is explicit that if a
segment's typical profile disagrees with a specific customer's actual
priority-matrix result, the priority matrix wins for that customer). It's
kept below as `SEGMENT_ACTION_TABLE`, a plain reference dict, not a
function -- there is nothing to "resolve" about it because it never
produces a conflicting per-customer decision.

Depends on:
  - `churn_model.py`'s predicted churn probability (thresholded at
    `config.CHURN_HIGH_RISK_THRESHOLD` into "High"/"Low" risk).
  - `ltv_model.py`'s predicted LTV tier (via
    `feature_engineering.assign_ltv_tier()` applied to LTV predictions --
    "High"/"Medium"/"Low").

Both are required inputs; this module does not compute either itself.
"""

import logging

import pandas as pd

from src.config import CHURN_HIGH_RISK_THRESHOLD

logger = logging.getLogger(__name__)

CHURN_RISK_LABELS = ("High", "Low")
LTV_TIER_LABELS = ("High", "Medium", "Low")

# The canonical, algorithmic per-customer logic (CLAUDE.md Section 6).
# Keyed by (churn_risk_label, ltv_tier) -> (priority_label, action).
# All 6 combinations (2 risk x 3 LTV tiers) are defined -- no fallthrough.
PRIORITY_MATRIX: dict[tuple[str, str], tuple[str, str]] = {
    ("High", "High"): (
        "Save Immediately",
        "Personalized retention offer + priority outreach",
    ),
    ("High", "Medium"): (
        "Targeted Retention",
        "Lighter-touch retention offer -- worth saving, not top priority spend",
    ),
    ("High", "Low"): (
        "Low Priority",
        "Minimal/no intervention -- cost of saving exceeds value",
    ),
    ("Low", "High"): (
        "Maintain Loyalty",
        "Loyalty rewards, upsell premium features",
    ),
    ("Low", "Medium"): (
        "Grow Engagement",
        "Standard engagement + light upsell nudges, watch for LTV growth",
    ),
    ("Low", "Low"): (
        "Monitor",
        "No action needed, standard engagement only",
    ),
}

# Narrative/strategic overlay only (CLAUDE.md Section 6) -- used for
# segment-level dashboard messaging and README narrative, e.g. "Champions
# get VIP rewards" as a segment-level story. NEVER used to override or
# recompute a specific customer's priority-matrix result; if a segment's
# typical profile disagrees with an individual customer's actual risk/LTV,
# the priority matrix (above) wins for that customer.
SEGMENT_ACTION_TABLE: dict[str, dict[str, str]] = {
    "Champions": {
        "typical_churn_risk": "Low",
        "typical_ltv": "High",
        "action": "VIP rewards, early access to new features",
    },
    "Loyal Customers": {
        "typical_churn_risk": "Low-Medium",
        "typical_ltv": "High",
        "action": "Upsell premium plan, loyalty program",
    },
    "At Risk": {
        "typical_churn_risk": "High",
        "typical_ltv": "High",
        "action": "Personalized retention offer, proactive outreach",
    },
    "New/Low Engagement": {
        "typical_churn_risk": "Medium",
        "typical_ltv": "Low-Medium",
        "action": "Onboarding nudges, engagement campaigns",
    },
    "Lost": {
        "typical_churn_risk": "High",
        "typical_ltv": "Low",
        "action": "Minimal marketing spend, deprioritize",
    },
}


# --------------------------------------------------------------------------
# Risk classification
# --------------------------------------------------------------------------

def classify_churn_risk(
    churn_probability: float, threshold: float = CHURN_HIGH_RISK_THRESHOLD
) -> str:
    """Threshold a churn probability into the "High"/"Low" risk label.

    Args:
        churn_probability: Predicted churn probability (e.g. from
            `churn_model.predict_churn_probability`).
        threshold: Cutoff for "High" risk. Defaults to
            `config.CHURN_HIGH_RISK_THRESHOLD` (0.5).

    Returns:
        "High" if `churn_probability >= threshold`, else "Low".
    """
    return "High" if churn_probability >= threshold else "Low"


# --------------------------------------------------------------------------
# Priority matrix lookup
# --------------------------------------------------------------------------

def get_priority(churn_risk: str, ltv_tier: str) -> tuple[str, str]:
    """Look up the priority label and action for a (risk, LTV tier) pair.

    Args:
        churn_risk: "High" or "Low" (see `classify_churn_risk`).
        ltv_tier: "High", "Medium", or "Low" (see
            `feature_engineering.assign_ltv_tier`).

    Returns:
        Tuple of (priority_label, action).

    Raises:
        KeyError: If the (churn_risk, ltv_tier) combination is not one of
            the 6 defined cells -- this should be unreachable given valid
            inputs, but fails loudly rather than silently falling through
            (CLAUDE.md Section 7 explicitly requires no undefined case).
    """
    key = (churn_risk, ltv_tier)
    if key not in PRIORITY_MATRIX:
        raise KeyError(
            f"No priority matrix entry for churn_risk={churn_risk!r}, "
            f"ltv_tier={ltv_tier!r}. Valid churn_risk values: "
            f"{CHURN_RISK_LABELS}; valid ltv_tier values: {LTV_TIER_LABELS}."
        )
    return PRIORITY_MATRIX[key]


# --------------------------------------------------------------------------
# Per-customer orchestration
# --------------------------------------------------------------------------

def build_recommendations(
    df: pd.DataFrame,
    churn_probability_col: str = "churn_probability",
    ltv_tier_col: str = "predicted_ltv_tier",
    threshold: float = CHURN_HIGH_RISK_THRESHOLD,
) -> pd.DataFrame:
    """Attach churn-risk label, priority label, and action to every row.

    This is the function the dashboard's Recommendations page (CLAUDE.md
    Section 9) calls directly -- the canonical per-customer table.

    Args:
        df: Dataframe with a predicted churn probability column and a
            predicted LTV tier column already attached (e.g. via
            `churn_model.predict_churn_probability` +
            `ltv_model.tier_predictions`).
        churn_probability_col: Column name holding predicted churn
            probabilities.
        ltv_tier_col: Column name holding predicted LTV tiers
            ("High"/"Medium"/"Low").
        threshold: Churn-risk cutoff, passed to `classify_churn_risk`.

    Returns:
        A copy of ``df`` with `Churn_Risk`, `Priority_Label`, and `Action`
        columns added.

    Raises:
        KeyError: If either required input column is missing, or if any
            row's `ltv_tier` value isn't one of the 3 valid tiers.
    """
    missing = {churn_probability_col, ltv_tier_col} - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns {sorted(missing)} -- run the churn and LTV "
            "models (and LTV tiering) before build_recommendations()."
        )

    df = df.copy()
    df["Churn_Risk"] = df[churn_probability_col].apply(
        lambda p: classify_churn_risk(p, threshold)
    )

    priorities = df.apply(
        lambda row: get_priority(row["Churn_Risk"], row[ltv_tier_col]), axis=1
    )
    df["Priority_Label"] = priorities.apply(lambda t: t[0])
    df["Action"] = priorities.apply(lambda t: t[1])

    logger.info(
        "Built recommendations for %d customers. Priority label counts:\n%s",
        len(df), df["Priority_Label"].value_counts().to_string(),
    )
    return df


def priority_matrix_as_dataframe() -> pd.DataFrame:
    """Render `PRIORITY_MATRIX` as a flat dataframe, for display/testing.

    Returns:
        Dataframe with `Churn_Risk`, `LTV_Tier`, `Priority_Label`, `Action`
        columns -- one row per of the 6 defined combinations.
    """
    rows = [
        {"Churn_Risk": risk, "LTV_Tier": tier, "Priority_Label": label, "Action": action}
        for (risk, tier), (label, action) in PRIORITY_MATRIX.items()
    ]
    return pd.DataFrame(rows)