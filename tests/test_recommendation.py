"""Tests for src/recommendations.py.

CLAUDE.md Section 7 explicitly requires: assert every combination of Churn
Risk x LTV tier, including Medium LTV, resolves to a defined priority label
with no fallthrough/undefined case.
"""

import itertools

import pandas as pd
import pytest

from src.recommendations import (
    CHURN_RISK_LABELS,
    LTV_TIER_LABELS,
    PRIORITY_MATRIX,
    build_recommendations,
    classify_churn_risk,
    get_priority,
    priority_matrix_as_dataframe,
)


class TestClassifyChurnRisk:
    def test_at_or_above_threshold_is_high(self):
        assert classify_churn_risk(0.5) == "High"
        assert classify_churn_risk(0.9) == "High"

    def test_below_threshold_is_low(self):
        assert classify_churn_risk(0.49) == "Low"
        assert classify_churn_risk(0.0) == "Low"

    def test_custom_threshold(self):
        assert classify_churn_risk(0.3, threshold=0.25) == "High"
        assert classify_churn_risk(0.2, threshold=0.25) == "Low"


class TestPriorityMatrixExhaustive:
    """Core required test (CLAUDE.md Section 7)."""

    @pytest.mark.parametrize(
        "risk,tier", list(itertools.product(CHURN_RISK_LABELS, LTV_TIER_LABELS))
    )
    def test_every_combination_resolves(self, risk, tier):
        label, action = get_priority(risk, tier)
        assert isinstance(label, str) and label
        assert isinstance(action, str) and action

    def test_matrix_has_exactly_six_defined_cells(self):
        assert len(PRIORITY_MATRIX) == len(CHURN_RISK_LABELS) * len(LTV_TIER_LABELS) == 6

    def test_medium_ltv_is_not_missing(self):
        """Explicit regression guard against the 4-row High/Low-only bug
        CLAUDE.md Section 6 calls out by name."""
        assert ("High", "Medium") in PRIORITY_MATRIX
        assert ("Low", "Medium") in PRIORITY_MATRIX

    def test_invalid_combination_raises(self):
        with pytest.raises(KeyError):
            get_priority("Medium", "High")  # "Medium" is not a valid churn risk label
        with pytest.raises(KeyError):
            get_priority("High", "Extreme")  # "Extreme" is not a valid LTV tier

    def test_matches_locked_claude_md_table(self):
        """Spot-check a few cells against CLAUDE.md Section 6's exact table."""
        assert get_priority("High", "High") == (
            "Save Immediately",
            "Personalized retention offer + priority outreach",
        )
        assert get_priority("Low", "Low") == (
            "Monitor",
            "No action needed, standard engagement only",
        )
        assert get_priority("High", "Low")[0] == "Low Priority"
        assert get_priority("Low", "Medium")[0] == "Grow Engagement"


class TestPriorityMatrixAsDataframe:
    def test_has_six_rows_and_expected_columns(self):
        df = priority_matrix_as_dataframe()
        assert len(df) == 6
        assert set(df.columns) == {"Churn_Risk", "LTV_Tier", "Priority_Label", "Action"}


class TestBuildRecommendations:
    @pytest.fixture
    def scored_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "customerID": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "churn_probability": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
            "predicted_ltv_tier": ["High", "Medium", "Low", "High", "Medium", "Low"],
        })

    def test_adds_expected_columns(self, scored_df):
        result = build_recommendations(scored_df)
        assert {"Churn_Risk", "Priority_Label", "Action"}.issubset(result.columns)
        assert len(result) == len(scored_df)

    def test_priority_labels_match_matrix(self, scored_df):
        result = build_recommendations(scored_df)
        expected_labels = [
            "Save Immediately", "Targeted Retention", "Low Priority",
            "Maintain Loyalty", "Grow Engagement", "Monitor",
        ]
        assert result["Priority_Label"].tolist() == expected_labels

    def test_no_nulls_in_output(self, scored_df):
        result = build_recommendations(scored_df)
        assert result[["Churn_Risk", "Priority_Label", "Action"]].notna().all().all()

    def test_raises_if_required_columns_missing(self):
        df = pd.DataFrame({"customerID": ["C1"], "churn_probability": [0.5]})
        with pytest.raises(KeyError):
            build_recommendations(df)  # missing predicted_ltv_tier

    def test_raises_on_invalid_ltv_tier_value(self):
        df = pd.DataFrame({
            "customerID": ["C1"],
            "churn_probability": [0.9],
            "predicted_ltv_tier": ["Extreme"],
        })
        with pytest.raises(KeyError):
            build_recommendations(df)