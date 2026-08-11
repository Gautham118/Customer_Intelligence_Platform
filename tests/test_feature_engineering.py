"""Tests for src/feature_engineering.py.

Covers the two areas CLAUDE.md Section 7 explicitly flags as most likely to
have silent logic bugs: RFM scoring and the future_ltv_12m target formula.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_ltv_target,
    add_rfm_scores,
    add_synthetic_behavioral_features,
    assign_ltv_tier,
    engineer_features,
)


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """A small synthetic-but-realistic sample spanning contract types."""
    n = 300
    rng = np.random.default_rng(0)
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.2]
    )
    internet = rng.choice(["Fiber optic", "DSL", "No"], size=n, p=[0.45, 0.35, 0.2])
    tenure = rng.integers(0, 73, size=n)
    monthly = np.round(rng.uniform(18, 119, size=n), 2)
    churn = rng.choice(["Yes", "No"], size=n, p=[0.265, 0.735])
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "tenure": tenure,
            "Contract": contracts,
            "InternetService": internet,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
            "Churn": churn,
        }
    )


# --------------------------------------------------------------------------
# RFM scoring
# --------------------------------------------------------------------------

class TestRFMScoring:
    def test_scores_are_in_1_to_5_range(self, raw_sample):
        df = add_synthetic_behavioral_features(raw_sample)
        df = add_rfm_scores(df)
        for col in ["R_score", "F_score", "M_score"]:
            assert df[col].between(1, 5).all(), f"{col} has values outside 1-5"

    def test_recency_is_scored_inversely(self, raw_sample):
        """Fewer days_since_last_activity should mean a HIGHER R_score."""
        df = add_synthetic_behavioral_features(raw_sample)
        df = add_rfm_scores(df)
        # Correlation between raw recency days and R_score should be negative.
        corr = df["days_since_last_activity"].corr(df["R_score"])
        assert corr < 0, f"expected negative correlation, got {corr}"

    def test_frequency_scored_directly(self, raw_sample):
        """More days_active_last_90d should mean a HIGHER F_score."""
        df = add_synthetic_behavioral_features(raw_sample)
        df = add_rfm_scores(df)
        corr = df["days_active_last_90d"].corr(df["F_score"])
        assert corr > 0, f"expected positive correlation, got {corr}"

    def test_rfm_score_is_three_digit_concatenation(self, raw_sample):
        df = add_synthetic_behavioral_features(raw_sample)
        df = add_rfm_scores(df)
        assert df["RFM_Score"].str.match(r"^[1-5]{3}$").all()
        # Spot check the concatenation is actually R+F+M in that order.
        row = df.iloc[0]
        expected = f"{row.R_score}{row.F_score}{row.M_score}"
        assert row["RFM_Score"] == expected

    def test_rfm_segment_has_no_nulls_or_unmapped_values(self, raw_sample):
        df = add_synthetic_behavioral_features(raw_sample)
        df = add_rfm_scores(df)
        assert df["RFM_Segment"].notna().all()
        assert (df["RFM_Segment"] != "").all()

    def test_raises_if_called_before_synthetic_features(self, raw_sample):
        with pytest.raises(KeyError):
            add_rfm_scores(raw_sample)


# --------------------------------------------------------------------------
# LTV target formula
# --------------------------------------------------------------------------

class TestLTVTarget:
    def test_ltv_is_non_negative(self, raw_sample):
        df = add_synthetic_behavioral_features(raw_sample)
        df = add_ltv_target(df)
        assert (df["future_ltv_12m"] >= 0).all()

    def test_not_a_deterministic_function_of_tenure_times_monthly_charges(self):
        """Core required test: identical inputs must NOT collapse to a
        deterministic future_ltv_12m -- the multiplier/noise must actually
        produce variation across otherwise-identical customers, otherwise
        the target is just a relabeled TotalCharges-style leak.
        """
        n = 200
        identical = pd.DataFrame(
            {
                "customerID": [f"C{i:04d}" for i in range(n)],
                "tenure": [24] * n,
                "Contract": ["One year"] * n,
                "InternetService": ["Fiber optic"] * n,
                "MonthlyCharges": [70.0] * n,
                "TotalCharges": [1680.0] * n,
                "Churn": ["No"] * n,
            }
        )
        df = add_synthetic_behavioral_features(identical)
        df = add_ltv_target(df)

        assert df["future_ltv_12m"].std() > 0, (
            "future_ltv_12m has zero variance across identical customers -- "
            "it has collapsed into a deterministic function of its inputs"
        )
        # Also confirm it isn't literally equal to tenure*MonthlyCharges*12
        # (the leakage pattern CLAUDE.md explicitly warns against).
        naive_leak_value = 24 * 70.0
        assert not np.isclose(df["future_ltv_12m"].mean(), naive_leak_value, rtol=0.05)

    def test_churned_customers_have_lower_mean_ltv_than_retained(self, raw_sample):
        """Directional sanity check on the retention_multiplier logic."""
        df = add_synthetic_behavioral_features(raw_sample)
        df = add_ltv_target(df)
        churned_mean = df.loc[df["Churn"] == "Yes", "future_ltv_12m"].mean()
        retained_mean = df.loc[df["Churn"] == "No", "future_ltv_12m"].mean()
        assert churned_mean < retained_mean

    def test_raises_if_required_columns_missing(self, raw_sample):
        with pytest.raises(KeyError):
            add_ltv_target(raw_sample)  # missing monthly_spend_trend_pct


class TestLTVTier:
    def test_exhaustive_and_no_unmapped_values(self):
        values = pd.Series(np.random.default_rng(1).uniform(0, 1000, size=500))
        tiers = assign_ltv_tier(values)
        assert tiers.notna().all()
        assert set(tiers.unique()) <= {"High", "Medium", "Low"}

    def test_roughly_matches_25_25_50_split(self):
        values = pd.Series(np.random.default_rng(1).uniform(0, 1000, size=2000))
        tiers = assign_ltv_tier(values)
        counts = tiers.value_counts(normalize=True)
        assert counts["High"] == pytest.approx(0.25, abs=0.02)
        assert counts["Medium"] == pytest.approx(0.25, abs=0.02)
        assert counts["Low"] == pytest.approx(0.50, abs=0.02)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_full_pipeline_runs_and_adds_expected_columns(self, raw_sample):
        df = engineer_features(raw_sample)
        expected_new_cols = {
            "avg_session_duration_min", "days_since_last_activity",
            "days_active_last_90d", "service_usage_interval_days",
            "monthly_spend_trend_pct", "avg_addon_spend",
            "weekend_activity_ratio", "R_score", "F_score", "M_score",
            "RFM_Score", "RFM_Segment", "future_ltv_12m", "future_ltv_12m_tier",
        }
        assert expected_new_cols.issubset(df.columns)
        assert len(df) == len(raw_sample)

    def test_reproducible_with_same_seed(self, raw_sample):
        df1 = engineer_features(raw_sample, random_state=42)
        df2 = engineer_features(raw_sample, random_state=42)
        pd.testing.assert_series_equal(df1["future_ltv_12m"], df2["future_ltv_12m"])