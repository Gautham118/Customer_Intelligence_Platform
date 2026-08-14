"""LTV Prediction -- model performance and per-customer lookup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import load_and_score_data, load_ltv_model, load_model_feature_matrices

st.set_page_config(page_title="LTV Prediction", page_icon="\U0001F4B0", layout="wide")
st.title("LTV Prediction")

st.caption(
    "XGBoost regressor predicting `future_ltv_12m`, a **disclosed synthetic "
    "proxy** target (the Telco dataset has no real forward-looking revenue "
    "field). The ground-truth `Churn` label is excluded from this model's "
    "features -- only the churn classifier's predicted probability is used "
    "as a churn-risk input, since a new customer being scored wouldn't "
    "have a true label yet either."
)

df, _ = load_and_score_data()
model = load_ltv_model()

c1, c2, c3 = st.columns(3)
c1.metric("Avg predicted LTV", f"${df['predicted_ltv'].mean():,.0f}")
c2.metric("High LTV customers", f"{(df['predicted_ltv_tier'] == 'High').sum():,}",
          f"{(df['predicted_ltv_tier'] == 'High').mean():.1%} of base")
c3.metric("Median predicted LTV", f"${df['predicted_ltv'].median():,.0f}")

st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Predicted LTV distribution")
    fig = px.histogram(df, x="predicted_ltv", color="predicted_ltv_tier", nbins=50,
                        category_orders={"predicted_ltv_tier": ["Low", "Medium", "High"]})
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Feature importance")
    _X_churn_unused, X = load_model_feature_matrices()
    importance = pd.DataFrame({
        "feature": X.columns, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(12)
    fig2 = px.bar(importance, x="importance", y="feature", orientation="h")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, width='stretch')

st.markdown("---")
st.subheader("LTV tier by segment")
tier_by_seg = (
    df.groupby(["Segment_Name", "predicted_ltv_tier"]).size().reset_index(name="count")
)
fig3 = px.bar(
    tier_by_seg, x="Segment_Name", y="count", color="predicted_ltv_tier",
    category_orders={"predicted_ltv_tier": ["Low", "Medium", "High"]}, barmode="stack",
)
st.plotly_chart(fig3, width='stretch')

st.markdown("---")
st.subheader("Look up a customer")
customer_id = st.selectbox("Customer ID", sorted(df["customerID"].unique()))
row = df.loc[df["customerID"] == customer_id].iloc[0]

lc1, lc2, lc3 = st.columns(3)
lc1.metric("Predicted 12mo LTV", f"${row['predicted_ltv']:,.0f}")
lc2.metric("LTV tier", row["predicted_ltv_tier"])
lc3.metric("Monthly charges", f"${row['MonthlyCharges']:.2f}")

st.markdown("---")
st.subheader("Highest predicted LTV customers")
st.dataframe(
    df.sort_values("predicted_ltv", ascending=False)
    [["customerID", "predicted_ltv", "predicted_ltv_tier", "Segment_Name",
      "churn_probability", "Priority_Label"]]
    .head(25),
    width='stretch', hide_index=True,
)

st.caption("`future_ltv_12m` is a disclosed synthetic proxy, not observed revenue.")