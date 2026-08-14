"""Recommendations -- the interactive Churn Risk x LTV priority matrix view.

CLAUDE.md Section 9: "The Churn Risk x LTV priority matrix is a real,
interactive table/view in the dashboard -- not just a static markdown
table in a notebook." This page is that requirement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import load_and_score_data
from src.recommendations import priority_matrix_as_dataframe, SEGMENT_ACTION_TABLE

st.set_page_config(page_title="Recommendations", page_icon="\u2705", layout="wide")
st.title("Recommendations")
st.caption(
    "The Churn Risk x LTV Priority Matrix is the canonical, per-customer "
    "recommendation logic -- it operates on each customer's actual "
    "predicted values. The Segment -> Action table below is a "
    "narrative/strategic overlay only; if a segment's typical profile ever "
    "disagrees with an individual customer's actual priority-matrix "
    "result, the priority matrix wins for that customer."
)

df, _ = load_and_score_data()

st.subheader("Priority matrix (reference)")
st.dataframe(priority_matrix_as_dataframe(), width='stretch', hide_index=True)

st.markdown("---")
st.subheader("Filter customers")

col1, col2, col3 = st.columns(3)
with col1:
    risk_filter = st.multiselect("Churn Risk", ["High", "Low"], default=["High", "Low"])
with col2:
    tier_filter = st.multiselect(
        "LTV Tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
    )
with col3:
    priority_filter = st.multiselect(
        "Priority Label", sorted(df["Priority_Label"].unique()),
        default=sorted(df["Priority_Label"].unique()),
    )

filtered = df[
    df["Churn_Risk"].isin(risk_filter)
    & df["predicted_ltv_tier"].isin(tier_filter)
    & df["Priority_Label"].isin(priority_filter)
]

st.caption(f"{len(filtered):,} of {len(df):,} customers match the current filters")

col_a, col_b = st.columns([1, 2])
with col_a:
    counts = filtered["Priority_Label"].value_counts().reset_index()
    counts.columns = ["Priority", "Customers"]
    fig = px.bar(counts, x="Customers", y="Priority", orientation="h", color="Priority")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, width='stretch')

with col_b:
    st.dataframe(
        filtered[[
            "customerID", "Segment_Name", "churn_probability", "predicted_ltv",
            "predicted_ltv_tier", "Churn_Risk", "Priority_Label", "Action",
        ]].sort_values("churn_probability", ascending=False),
        width='stretch', hide_index=True, height=420,
    )

st.download_button(
    "Download filtered recommendations (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="customer_recommendations.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("Segment -> Action overlay (narrative reference, not per-customer logic)")
st.dataframe(
    pd.DataFrame(SEGMENT_ACTION_TABLE).T,
    width='stretch',
)