"""RFM derivation, synthetic behavioral features, and the LTV target proxy.

Prerequisites for all downstream modeling per CLAUDE.md Section 9: this file
must run before churn_model.py, ltv_model.py, and segmentation.py, since they
all depend on its outputs.

Leakage guardrail (CLAUDE.md Section 6): every synthetic behavioral feature
is generated as a function of tenure, MonthlyCharges, Contract, and
InternetService only. `Churn` is NEVER used to construct a feature -- it is
only used afterward, in `log_synthetic_feature_sanity_checks`, to confirm the
generated distributions look directionally plausible. The LTV target
(`future_ltv_12m`) is the one deliberate exception: it explicitly uses
`Churn` to build the *target*, which is a different thing from using it as a
*feature* -- see `add_ltv_target` docstring for why that's fine.
"""

import logging

import numpy as np
import pandas as pd

from src.config import LTV_HIGH_PERCENTILE, LTV_MEDIUM_PERCENTILE, RANDOM_STATE, RFM_QUANTILES

logger = logging.getLogger(__name__)

CONTRACT_COMMITMENT_WEIGHT = {"Month-to-month": 0.2, "One year": 0.6, "Two year": 1.0}

SYNTHETIC_BEHAVIORAL_COLUMNS = [
    "avg_session_duration_min",
    "days_since_last_activity",
    "days_active_last_90d",
    "service_usage_interval_days",
    "monthly_spend_trend_pct",
    "avg_addon_spend",
    "weekend_activity_ratio",
]


# --------------------------------------------------------------------------
# Synthetic behavioral features
# --------------------------------------------------------------------------

def _engagement_score(df: pd.DataFrame) -> np.ndarray:
    """Latent 0-1 'engagement' score driving several synthetic features.

    Built only from tenure (percentile rank) and Contract commitment
    (Month-to-month=0.2, One year=0.6, Two year=1.0), averaged evenly.
    Higher = longer-tenured, more contractually committed customer. This is
    an internal helper, not a column added to the dataframe -- it just keeps
    the per-feature formulas below short and gives them one consistent,
    explainable story.

    Args:
        df: Dataframe with `tenure` and `Contract` columns.

    Returns:
        Array of engagement scores in [0, 1], one per row.
    """
    tenure_pct = df["tenure"].rank(pct=True).to_numpy()
    contract_weight = df["Contract"].map(CONTRACT_COMMITMENT_WEIGHT).to_numpy()
    return 0.5 * tenure_pct + 0.5 * contract_weight


def add_synthetic_behavioral_features(
    df: pd.DataFrame, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Add 7 synthetic behavioral columns simulating app/portal activity.

    Every feature is generated purely from `tenure`, `Contract`,
    `MonthlyCharges`, and `InternetService` (real columns) plus a fixed-seed
    random generator -- never from `Churn`. See module docstring.

    Args:
        df: Cleaned dataframe (post `preprocessing.clean_data`).
        random_state: Seed for the synthetic RNG, for reproducibility.

    Returns:
        A copy of ``df`` with 7 new synthetic columns added (see
        ``SYNTHETIC_BEHAVIORAL_COLUMNS``).
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)
    n = len(df)

    engagement = _engagement_score(df)
    contract_weight = df["Contract"].map(CONTRACT_COMMITMENT_WEIGHT).to_numpy()
    tenure_days = df["tenure"].to_numpy() * 30

    # avg_session_duration_min: lognormal, median rises with engagement.
    median_minutes = 5 + engagement * 15
    session = rng.lognormal(mean=np.log(median_minutes), sigma=0.4)
    df["avg_session_duration_min"] = np.clip(session, 1, 120).round(1)

    # days_since_last_activity: exponential, more engaged -> smaller scale.
    # Capped so a brand-new customer can't have a "last activity" older
    # than their own account.
    scale = 2 + (1 - engagement) * 20
    days_since = rng.exponential(scale=scale)
    df["days_since_last_activity"] = np.minimum(days_since, tenure_days + 1).round(1)

    # days_active_last_90d: Binomial(window, p). window is capped by tenure
    # so customers who haven't existed for 90 days aren't credited with a
    # full 90-day activity history.
    window = np.minimum(90, tenure_days).astype(int)
    p_active = np.clip(0.1 + engagement * 0.6, 0, 1)
    df["days_active_last_90d"] = rng.binomial(n=window, p=p_active)

    # service_usage_interval_days: gamma, same directional driver as the two
    # features above but a different distribution shape, so it's correlated
    # with them (as real behavioral data would be) without being a
    # deterministic transform of either.
    mean_interval = 3 + (1 - engagement) * 15
    df["service_usage_interval_days"] = rng.gamma(
        shape=2.0, scale=mean_interval / 2.0
    ).round(1)

    # monthly_spend_trend_pct: normal, centered near 0. Tighter variance for
    # longer/locked-in contracts (more stable billing).
    mean_trend = (engagement - 0.5) * 4
    std_trend = 8 - contract_weight * 4
    trend = rng.normal(mean_trend, std_trend)
    df["monthly_spend_trend_pct"] = np.clip(trend, -50, 50).round(2)

    # avg_addon_spend: fraction of MonthlyCharges, boosted for customers
    # with internet service (more add-ons available to buy).
    internet_bonus = df["InternetService"].map(
        {"Fiber optic": 0.05, "DSL": 0.02, "No": 0.0}
    ).to_numpy()
    addon_frac = 0.05 + engagement * 0.15 + internet_bonus
    addon_noise = rng.lognormal(mean=0, sigma=0.3, size=n)
    df["avg_addon_spend"] = (
        df["MonthlyCharges"].to_numpy() * addon_frac * addon_noise
    ).round(2)

    # weekend_activity_ratio: Beta, mean shifts slightly by contract
    # commitment (month-to-month customers skew a bit more weekend-heavy).
    # Beta's natural [0,1] support avoids any post-hoc clipping.
    mean_ratio = 0.32 - contract_weight * 0.08
    concentration = 10
    df["weekend_activity_ratio"] = rng.beta(
        mean_ratio * concentration, (1 - mean_ratio) * concentration
    ).round(3)

    logger.info("Added %d synthetic behavioral features for %d rows",
                len(SYNTHETIC_BEHAVIORAL_COLUMNS), n)
    return df


def log_synthetic_feature_sanity_checks(df: pd.DataFrame) -> None:
    """Log directional sanity checks of synthetic features against Churn.

    This is the ONLY place `Churn` may touch the synthetic features -- for
    inspection after the fact, never for generation (CLAUDE.md Section 6).
    Purely logs group means; does not modify or return anything.

    Args:
        df: Dataframe with synthetic behavioral features and `Churn` present.
    """
    check_cols = [
        "days_since_last_activity", "days_active_last_90d",
        "avg_session_duration_min", "weekend_activity_ratio",
    ]
    means = df.groupby("Churn")[check_cols].mean().round(2)
    logger.info("Synthetic feature sanity check (mean by Churn):\n%s", means)


# --------------------------------------------------------------------------
# RFM scoring
# --------------------------------------------------------------------------

def _qcut_score(
    series: pd.Series, ascending: bool, n_quantiles: int = RFM_QUANTILES
) -> pd.Series:
    """Score a numeric series 1..n_quantiles by quintile.

    Uses `pd.qcut(duplicates="drop")` first. If tied bin edges collapse the
    result below `n_quantiles` bins (common for low-variance columns), falls
    back to percentile-rank bucketing instead of raising, per CLAUDE.md
    Section 6.

    Args:
        series: Numeric column to score.
        ascending: If True, higher raw values get higher scores (standard).
            If False, LOWER raw values get higher scores -- used for
            Recency, where fewer days-since-activity should score higher.
        n_quantiles: Number of buckets (5 for standard RFM quintiles).

    Returns:
        Integer Series with values in [1, n_quantiles].
    """
    try:
        binned = pd.qcut(series, q=n_quantiles, duplicates="drop")
        n_bins = binned.cat.categories.size
        if n_bins < n_quantiles:
            raise ValueError(f"qcut collapsed to {n_bins} bins (< {n_quantiles})")
        scores = binned.cat.codes.to_numpy() + 1
    except ValueError as e:
        logger.warning(
            "qcut fallback triggered for column '%s' (%s); using "
            "rank(pct=True) bucketing instead", series.name, e,
        )
        pct_rank = series.rank(pct=True, method="average").to_numpy()
        scores = np.ceil(pct_rank * n_quantiles).clip(1, n_quantiles).astype(int)

    scores = pd.Series(scores, index=series.index)
    if not ascending:
        scores = (n_quantiles + 1) - scores
    return scores.astype(int)


def _rfm_segment(row: pd.Series) -> str:
    """Map an (R, F, M) score triple to a coarse RFM segment label.

    This is a standard, simple RFM heuristic -- NOT one of CLAUDE.md's
    locked thresholds (only the K-Means cluster-naming taxonomy in Section 6
    is locked). It reuses the same five label names for narrative
    consistency, but is computed independently of the K-Means clusters built
    later in segmentation.py.
    """
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if f >= 4 and m >= 3:
        return "Loyal Customers"
    if r <= 2 and (f >= 3 or m >= 3):
        return "At Risk"
    if r >= 4 and f <= 2:
        return "New/Low Engagement"
    if r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    return "Needs Attention"


def add_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Recency, Frequency, Monetary scores, RFM_Score, and RFM_Segment.

    Must be called AFTER `add_synthetic_behavioral_features`, since Recency
    and Frequency are derived from synthetic columns (CLAUDE.md Section 6):
      - Recency   <- days_since_last_activity (inverted: fewer days = higher score)
      - Frequency <- days_active_last_90d
      - Monetary  <- MonthlyCharges

    Args:
        df: Dataframe that already has the synthetic behavioral columns.

    Returns:
        A copy of ``df`` with ``R_score``, ``F_score``, ``M_score`` (1-5
        each), ``RFM_Score`` (3-digit string, e.g. "543"), and
        ``RFM_Segment`` (label) added.

    Raises:
        KeyError: If synthetic behavioral columns are missing (i.e. this
            was called before `add_synthetic_behavioral_features`).
    """
    required = {"days_since_last_activity", "days_active_last_90d", "MonthlyCharges"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns {sorted(missing)} -- call "
            "add_synthetic_behavioral_features(df) before add_rfm_scores(df)."
        )

    df = df.copy()
    df["R_score"] = _qcut_score(df["days_since_last_activity"], ascending=False)
    df["F_score"] = _qcut_score(df["days_active_last_90d"], ascending=True)
    df["M_score"] = _qcut_score(df["MonthlyCharges"], ascending=True)
    df["RFM_Score"] = (
        df["R_score"].astype(str) + df["F_score"].astype(str) + df["M_score"].astype(str)
    )
    df["RFM_Segment"] = df.apply(_rfm_segment, axis=1)

    logger.info(
        "RFM scoring complete. Segment counts:\n%s",
        df["RFM_Segment"].value_counts().to_string(),
    )
    return df


# --------------------------------------------------------------------------
# LTV target proxy
# --------------------------------------------------------------------------

def add_ltv_target(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Build the `future_ltv_12m` LTV regression target (a disclosed proxy).

    future_ltv_12m = (MonthlyCharges x 12) x retention_multiplier + noise

    IMPORTANT distinction (CLAUDE.md Section 6): this function uses `Churn`
    to build the retention_multiplier -- that's fine and expected, since
    Churn describes what actually happened to the customer's future
    relationship, which is exactly what the *target* needs to reflect. This
    is different from using Churn as a *feature*. When ltv_model.py selects
    its training features from this dataframe, `Churn` must be excluded
    from that feature set -- only the churn classifier's predicted
    probability may be used as an LTV input, never the ground-truth label.
    That exclusion happens in ltv_model.py, not here.

    retention_multiplier:
      - Churned customers: Uniform(0.05, 0.35) -- their relationship ended
        partway through the 12-month window, so realized future spend is
        heavily truncated.
      - Retained customers: clip(0.85 + contract_weight*0.15 + trend_adj,
        0.5, 1.3) -- more contractually committed and/or growing-spend
        customers get a multiplier above 1.0 (mild growth); declining-spend
        customers stay closer to 0.85.
    noise: Gaussian, std = 7.5% of the base value (MonthlyCharges*12*
      multiplier) -- the midpoint of CLAUDE.md's "5-10% of base value" spec.

    Args:
        df: Dataframe that already has `monthly_spend_trend_pct` (i.e.
            called after `add_synthetic_behavioral_features`) and `Churn`.
        random_state: Seed for the multiplier/noise RNG.

    Returns:
        A copy of ``df`` with `future_ltv_12m` added (float, >= 0).

    Raises:
        KeyError: If required columns are missing.
    """
    required = {"monthly_spend_trend_pct", "Churn", "Contract", "MonthlyCharges"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} for LTV target construction.")

    df = df.copy()
    rng = np.random.default_rng(random_state)
    n = len(df)

    contract_weight = df["Contract"].map(CONTRACT_COMMITMENT_WEIGHT).to_numpy()
    trend_adj = (df["monthly_spend_trend_pct"].to_numpy() / 100) * 0.5
    is_churned = (df["Churn"] == "Yes").to_numpy()

    retained_mult = np.clip(0.85 + contract_weight * 0.15 + trend_adj, 0.5, 1.3)
    churned_mult = rng.uniform(0.05, 0.35, size=n)
    multiplier = np.where(is_churned, churned_mult, retained_mult)

    base = df["MonthlyCharges"].to_numpy() * 12 * multiplier
    noise = rng.normal(0, base * 0.075)
    df["future_ltv_12m"] = np.clip(base + noise, 0, None).round(2)

    logger.info(
        "future_ltv_12m built (SYNTHETIC PROXY, not observed revenue). "
        "mean=%.2f, median=%.2f",
        df["future_ltv_12m"].mean(), df["future_ltv_12m"].median(),
    )
    return df


def assign_ltv_tier(values: pd.Series) -> pd.Series:
    """Bucket LTV values into the three locked tiers (CLAUDE.md Section 6).

    High = top 25% (>= 75th percentile)
    Medium = middle 25% (50th-74th percentile)
    Low = bottom 50% (< 50th percentile)
    Exhaustive: every value falls into exactly one tier.

    Reusable on ANY array of LTV values -- called here on the synthetic
    `future_ltv_12m` for early EDA, and again in ltv_model.py on the trained
    regressor's *predicted* values, which is what "LTV tier" actually means
    per CLAUDE.md ("predicted LTV values in the dataset").

    Args:
        values: Numeric LTV values (synthetic target or model predictions).

    Returns:
        String Series with values in {"High", "Medium", "Low"}.
    """
    p50 = values.quantile(LTV_MEDIUM_PERCENTILE)
    p75 = values.quantile(LTV_HIGH_PERCENTILE)
    tiers = pd.Series("Low", index=values.index)
    tiers[values >= p50] = "Medium"
    tiers[values >= p75] = "High"
    return tiers


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Run the full feature engineering pipeline in the required order.

    Order matters (CLAUDE.md Section 6/9): synthetic behavioral features
    must exist before RFM scores (R/F derive from them) and before the LTV
    target (which uses monthly_spend_trend_pct).

    Args:
        df: Cleaned dataframe (post `preprocessing.clean_data`).
        random_state: Seed shared by both synthetic-generation steps.

    Returns:
        Dataframe with synthetic behavioral features, RFM scores/segment,
        `future_ltv_12m`, and `future_ltv_12m_tier` (EDA-only tier based on
        the synthetic target -- re-tier on real predictions in ltv_model.py).
    """
    df = add_synthetic_behavioral_features(df, random_state=random_state)
    df = add_rfm_scores(df)
    df = add_ltv_target(df, random_state=random_state)
    df["future_ltv_12m_tier"] = assign_ltv_tier(df["future_ltv_12m"])
    log_synthetic_feature_sanity_checks(df)
    return df