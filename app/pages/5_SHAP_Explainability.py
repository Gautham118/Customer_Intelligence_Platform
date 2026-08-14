"""SHAP Explainability -- global feature importance + per-customer narratives.

CLAUDE.md Section 9 requires SHAP explanations for at least 3 example
customers with human-readable narrative text, not just raw plots. This page
makes that interactive: pick any customer, get their narrative + a
contribution chart, rather than 3 fixed static examples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import (
    load_and_score_data,
    load_churn_model,
    load_ltv_model,
    load_model_feature_matrices,
)
from src.explainability import (
    get_shap_explainer,
    compute_shap_values,
    global_feature_importance,
    generate_customer_narrative,
)

st.set_page_config(page_title="SHAP Explainability", page_icon="\U0001F50D", layout="wide")
st.title("SHAP Explainability")
st.caption(
    "SHAP TreeExplainer values for both models. Categorical features "
    "(native XGBoost categorical dtype) don't get a color gradient on the "
    "beeswarm-style ranking below -- this is a SHAP plotting limitation "
    "for mixed categorical/numeric feature sets, not an error in the "
    "underlying values."
)

df, _ = load_and_score_data()

model_choice = st.radio("Model", ["Churn", "LTV"], horizontal=True)


@st.cache_resource(show_spinner="Computing SHAP values (churn model)...")
def _churn_shap():
    X_churn, _X_ltv = load_model_feature_matrices()
    model = load_churn_model()
    explainer = get_shap_explainer(model)
    shap_values = compute_shap_values(explainer, X_churn)
    return model, explainer, X_churn, shap_values


@st.cache_resource(show_spinner="Computing SHAP values (LTV model)...")
def _ltv_shap():
    _X_churn, X_ltv = load_model_feature_matrices()
    model = load_ltv_model()
    explainer = get_shap_explainer(model)
    shap_values = compute_shap_values(explainer, X_ltv)
    return model, explainer, X_ltv, shap_values


if model_choice == "Churn":
    model, explainer, X, shap_values = _churn_shap()
    is_classifier = True
    pred_col = "churn_probability"
else:
    model, explainer, X, shap_values = _ltv_shap()
    is_classifier = False
    pred_col = "predicted_ltv"

st.subheader(f"Global feature importance ({model_choice} model)")
importance = global_feature_importance(shap_values, X)
fig = px.bar(
    importance.head(15), x="mean_abs_shap", y="feature", orientation="h",
    title="Mean |SHAP value| across all customers",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, width='stretch')

st.markdown("---")
st.subheader("Explain a customer")

customer_id = st.selectbox("Customer ID", df["customerID"].loc[X.index].tolist())
row_position = X.index.get_loc(df.index[df["customerID"] == customer_id][0])

expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[-1]

prediction = float(df.loc[X.index[row_position], pred_col])

narrative = generate_customer_narrative(
    X_row=X.iloc[row_position],
    shap_row=shap_values[row_position],
    base_value=float(expected_value),
    prediction=prediction,
    customer_id=customer_id,
    is_classifier=is_classifier,
)
st.info(narrative, icon="\U0001F4AC")

contrib = pd.DataFrame({
    "feature": X.columns,
    "shap_value": shap_values[row_position],
    "feature_value": X.iloc[row_position].astype(str).to_numpy(),
}).reindex(np.argsort(-np.abs(shap_values[row_position]))).head(15)
contrib["label"] = contrib["feature"] + " = " + contrib["feature_value"]

fig2 = px.bar(
    contrib, x="shap_value", y="label", orientation="h",
    color="shap_value", color_continuous_scale=["#3B82F6", "#EF4444"],
    title=f"Top contributing features for {customer_id}",
)
fig2.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
st.plotly_chart(fig2, width='stretch')