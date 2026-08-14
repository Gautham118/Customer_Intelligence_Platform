"""Executive Overview -- top-level KPIs and business narrative."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px
import streamlit as st

from src.utils import load_and_score_data

st.set_page_config(page_title="Executive Overview", page_icon="\U0001F4CA", layout="wide")
st.title("Executive Overview")

df, cluster_profiles = load_and_score_data()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total customers", f"{len(df):,}")
c2.metric("Churn rate", f"{(df['Churn'] == 'Yes').mean():.1%}")
c3.metric("High churn risk", f"{(df['Churn_Risk'] == 'High').sum():,}")
c4.metric("Avg predicted LTV", f"${df['predicted_ltv'].mean():,.0f}")
c5.metric("Save-Immediately customers", f"{(df['Priority_Label'] == 'Save Immediately').sum():,}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn risk vs. LTV tier")
    heat = (
        df.groupby(["Churn_Risk", "predicted_ltv_tier"])
        .size()
        .reset_index(name="count")
    )
    fig = px.density_heatmap(
        heat, x="predicted_ltv_tier", y="Churn_Risk", z="count",
        category_orders={"predicted_ltv_tier": ["Low", "Medium", "High"], "Churn_Risk": ["Low", "High"]},
        text_auto=True, color_continuous_scale="Blues",
    )
    fig.update_layout(xaxis_title="Predicted LTV Tier", yaxis_title="Churn Risk")
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Customers by segment")
    seg_counts = df["Segment_Name"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    fig2 = px.bar(seg_counts, x="Segment", y="Customers", color="Segment")
    st.plotly_chart(fig2, width='stretch')

st.markdown("---")
st.subheader("Priority action breakdown")
priority_summary = (
    df.groupby("Priority_Label")
    .agg(customers=("customerID", "count"), avg_ltv=("predicted_ltv", "mean"),
         avg_churn_prob=("churn_probability", "mean"))
    .round(2)
    .sort_values("customers", ascending=False)
)
st.dataframe(priority_summary, width='stretch')

st.caption(
    "Note: several behavioral features and the 12-month LTV target are "
    "synthetically simulated -- see README for full disclosure."
)