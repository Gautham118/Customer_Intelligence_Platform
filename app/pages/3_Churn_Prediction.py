"""Churn Prediction -- model performance and per-customer lookup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import load_and_score_data, load_churn_model, load_model_feature_matrices
from src.config import CHURN_HIGH_RISK_THRESHOLD

st.set_page_config(page_title="Churn Prediction", page_icon="\U0001F6A8", layout="wide")
st.title("Churn Prediction")

st.caption(
    "XGBoost classifier, scale_pos_weight recomputed per training fold. "
    "Optimized for Recall / ROC-AUC / PR-AUC, not raw accuracy, since the "
    "cost of missing a churner outweighs a wasted retention offer on a "
    "false positive. Predicted probabilities are not calibrated -- treat "
    f"the {CHURN_HIGH_RISK_THRESHOLD:.0%} threshold as a ranking cutoff, "
    "not a literal probability."
)

df, _ = load_and_score_data()
model = load_churn_model()

c1, c2, c3 = st.columns(3)
c1.metric("High risk customers", f"{(df['Churn_Risk'] == 'High').sum():,}",
          f"{(df['Churn_Risk'] == 'High').mean():.1%} of base")
c2.metric("Mean predicted probability", f"{df['churn_probability'].mean():.1%}")
c3.metric("Actual churn rate", f"{(df['Churn'] == 'Yes').mean():.1%}")

st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Predicted probability distribution")
    fig = px.histogram(
        df, x="churn_probability", color="Churn", nbins=40, barmode="overlay",
        opacity=0.7,
    )
    fig.add_vline(x=CHURN_HIGH_RISK_THRESHOLD, line_dash="dash",
                   annotation_text="High-risk threshold")
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Feature importance")
    X, _X_ltv_unused = load_model_feature_matrices()
    importance = pd.DataFrame({
        "feature": X.columns, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(12)
    fig2 = px.bar(importance, x="importance", y="feature", orientation="h")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, width='stretch')

st.markdown("---")
st.subheader("Look up a customer")
customer_id = st.selectbox("Customer ID", sorted(df["customerID"].unique()))
row = df.loc[df["customerID"] == customer_id].iloc[0]

lc1, lc2, lc3, lc4 = st.columns(4)
lc1.metric("Churn probability", f"{row['churn_probability']:.1%}")
lc2.metric("Risk flag", row["Churn_Risk"])
lc3.metric("Actual outcome", row["Churn"])
lc4.metric("Contract", row["Contract"])

st.dataframe(
    row[["tenure", "MonthlyCharges", "Contract", "InternetService", "RFM_Score",
         "RFM_Segment", "Segment_Name", "days_since_last_activity",
         "days_active_last_90d"]].astype(str).to_frame("value"),
    width='stretch',
)

st.markdown("---")
st.subheader("Highest-risk customers")
st.dataframe(
    df.sort_values("churn_probability", ascending=False)
    [["customerID", "churn_probability", "Segment_Name", "predicted_ltv", "Priority_Label"]]
    .head(25),
    width='stretch', hide_index=True,
)