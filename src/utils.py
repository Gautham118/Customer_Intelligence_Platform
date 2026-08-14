"""Shared helpers: logging setup, and cached pipeline/model loading for the
Streamlit dashboard.

CLAUDE.md Section 3 requires the dashboard to load pre-trained `.joblib`
model artifacts and import helpers from `src/` -- it must NOT re-run
training live. `load_and_score_data()` below is the single place that
enforces this: it loads already-saved models via `churn_model.load_model()`
/ `ltv_model.load_model()`, and only *predicts* with them, never calls
`train_final_model()`.

Streamlit's `st.cache_resource` / `st.cache_data` decorators are used here
so this expensive pipeline (raw data -> clean -> feature engineer ->
segment -> score) runs once per app session and is shared across all 6
dashboard pages, rather than re-running on every page navigation.
"""

import logging

import joblib
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR
from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.segmentation import segment_customers
from src.churn_model import (
    load_model as load_churn_model_file,
    prepare_features as churn_prepare_features,
    predict_churn_probability,
)
from src.ltv_model import (
    load_model as load_ltv_model_file,
    add_churn_probability_feature,
    prepare_features as ltv_prepare_features,
    predict_ltv,
    tier_predictions,
)
from src.recommendations import build_recommendations

logger = logging.getLogger(__name__)

SEGMENTATION_K = 5  # locked choice per DECISIONS.md -- do not leave k=None here


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the dashboard process."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


@st.cache_resource(show_spinner="Loading trained models...")
def load_churn_model():
    """Load the saved churn classifier (does not train)."""
    return load_churn_model_file(MODELS_DIR / "churn_model.joblib")


@st.cache_resource(show_spinner="Loading trained models...")
def load_ltv_model():
    """Load the saved LTV regressor (does not train)."""
    return load_ltv_model_file(MODELS_DIR / "ltv_model.joblib")


@st.cache_data(show_spinner="Running feature engineering + segmentation pipeline...")
def load_pipeline_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the (fast, non-training) data pipeline through segmentation.

    This is feature engineering + K-Means segmentation, NOT model training
    -- both are deterministic and cheap relative to XGBoost training, so
    re-running them at dashboard startup is fine per CLAUDE.md; only the
    two XGBoost models themselves must be loaded pre-trained.

    Returns:
        Tuple of (feature-engineered + segmented dataframe, cluster profile
        summary dataframe).
    """
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    df, cluster_profiles = segment_customers(df, k=SEGMENTATION_K)
    return df, cluster_profiles


@st.cache_data(show_spinner="Scoring customers (churn risk, LTV, recommendations)...")
def load_and_score_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full dashboard dataset: pipeline output + model predictions + recommendations.

    Loads the two pre-trained models (never trains them here), scores every
    customer, and attaches the canonical Churn Risk x LTV priority matrix
    recommendation. This is the single function every dashboard page should
    call to get its working dataframe for DISPLAY purposes.

    IMPORTANT: the returned dataframe has `churn_probability`,
    `predicted_ltv`, `predicted_ltv_tier`, `Churn_Risk`, `Priority_Label`,
    and `Action` columns attached -- do NOT re-run
    `churn_model.prepare_features()` / `ltv_model.prepare_features()` on
    this dataframe to rebuild a feature matrix (e.g. for SHAP or importance
    plots). Those derived columns aren't in either model's
    `EXCLUDED_COLUMNS` list (they didn't exist yet when that list was
    written), so re-deriving X from this df would leak them back in as if
    they were input features -- for the churn model, that would mean the
    model's own prediction leaking in as a "feature" when explaining it.
    Use `load_model_feature_matrices()` below instead, which builds both
    matrices once at the correct pre-scoring pipeline stage.

    Returns:
        Tuple of (fully scored dataframe, cluster profile summary dataframe).
    """
    df, cluster_profiles = load_pipeline_data()

    churn_model = load_churn_model()
    ltv_model = load_ltv_model()

    X_churn, _ = churn_prepare_features(df)
    df["churn_probability"] = predict_churn_probability(churn_model, X_churn)

    df_for_ltv = add_churn_probability_feature(df, df["churn_probability"].to_numpy())
    X_ltv, _ = ltv_prepare_features(df_for_ltv)
    df["predicted_ltv"] = predict_ltv(ltv_model, X_ltv)
    df["predicted_ltv_tier"] = tier_predictions(
        df["predicted_ltv"].to_numpy(), index=df.index
    )

    df = build_recommendations(df)

    logger.info("Scored %d customers for dashboard display", len(df))
    return df, cluster_profiles


@st.cache_data(show_spinner="Preparing model feature matrices...")
def load_model_feature_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (X_churn, X_ltv) once, at the correct pre-scoring pipeline stage.

    This is the single source of truth for "what did the model actually
    see as input" -- used by the Churn/LTV Prediction pages' feature
    importance charts and the SHAP Explainability page, instead of those
    pages re-deriving a feature matrix from the fully-scored display
    dataframe (see `load_and_score_data()` docstring for why that's unsafe).

    Returns:
        Tuple of (X_churn, X_ltv), row-aligned to `load_pipeline_data()`'s
        dataframe (i.e. NOT the display dataframe from
        `load_and_score_data()`, though both share the same row index).
    """
    df, _ = load_pipeline_data()
    churn_model = load_churn_model()

    X_churn, _ = churn_prepare_features(df)
    churn_proba = predict_churn_probability(churn_model, X_churn)

    df_for_ltv = add_churn_probability_feature(df, churn_proba)
    X_ltv, _ = ltv_prepare_features(df_for_ltv)

    return X_churn, X_ltv