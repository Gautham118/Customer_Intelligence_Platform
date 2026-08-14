"""Segments -- K-Means cluster profiles and taxonomy mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px
import streamlit as st

from src.utils import load_and_score_data

st.set_page_config(page_title="Segments", page_icon="\U0001F465", layout="wide")
st.title("Customer Segments")
st.caption(
    "K-Means clustering on R/F/M quintile scores. Cluster names are mapped "
    "to the nearest of 5 taxonomy archetypes by Euclidean distance; a "
    "cluster whose nearest archetype is too far gets flagged 'Unmapped' "
    "rather than forced into a label."
)

df, cluster_profiles = load_and_score_data()

st.subheader("Cluster profiles")
display_profiles = cluster_profiles.copy()
st.dataframe(display_profiles, width='stretch')

if display_profiles["ambiguous"].any():
    st.warning(
        f"{int(display_profiles['ambiguous'].sum())} cluster(s) did not clearly "
        "match any taxonomy archetype (match distance beyond threshold) -- "
        "review these manually rather than trusting the auto-assigned label.",
        icon="\u26a0\ufe0f",
    )

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Segment sizes")
    seg_counts = df["Segment_Name"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    fig = px.pie(seg_counts, names="Segment", values="Customers", hole=0.4)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Churn rate by segment")
    churn_by_seg = (
        df.groupby("Segment_Name")["Churn"]
        .apply(lambda s: (s == "Yes").mean())
        .reset_index(name="churn_rate")
        .sort_values("churn_rate", ascending=False)
    )
    fig2 = px.bar(churn_by_seg, x="Segment_Name", y="churn_rate", color="Segment_Name")
    fig2.update_layout(yaxis_tickformat=".0%", xaxis_title="Segment", yaxis_title="Churn rate")
    st.plotly_chart(fig2, width='stretch')

st.markdown("---")
st.subheader("Explore a segment")
selected_segment = st.selectbox("Segment", sorted(df["Segment_Name"].unique()))
segment_df = df[df["Segment_Name"] == selected_segment]

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Customers", f"{len(segment_df):,}")
sc2.metric("Churn rate", f"{(segment_df['Churn'] == 'Yes').mean():.1%}")
sc3.metric("Avg tenure", f"{segment_df['tenure'].mean():.1f} mo")
sc4.metric("Avg predicted LTV", f"${segment_df['predicted_ltv'].mean():,.0f}")

st.dataframe(
    segment_df[["customerID", "tenure", "Contract", "MonthlyCharges", "RFM_Score",
                "churn_probability", "predicted_ltv", "Priority_Label"]].head(50),
    width='stretch', hide_index=True,
)