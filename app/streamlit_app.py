"""Customer Intelligence Platform -- main dashboard entrypoint.

Run with: streamlit run app/streamlit_app.py

Per CLAUDE.md Section 3's v1 priority note: correctness/completeness over
visual styling for v1. This is intentionally plain default Streamlit
styling -- polish is a later pass.
"""

import sys
from pathlib import Path

# Make `from src...` importable when launched as `streamlit run app/streamlit_app.py`
# without relying on `pip install -e .` having been run in this exact shell.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.utils import load_and_score_data, setup_logging

setup_logging()

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="\U0001F4CA",
    layout="wide",
)

st.title("Customer Intelligence Platform")
st.markdown(
    "**Can we identify high-value customers who are likely to churn and "
    "recommend the most profitable retention strategy for each customer "
    "segment?**"
)

st.info(
    "Some behavioral features (session activity, spend trend, etc.) and the "
    "12-month LTV target are **synthetically simulated** for demonstration -- "
    "the Telco dataset itself has no native transaction log or forward-looking "
    "revenue field. See the README for full disclosure of what's real vs. "
    "simulated in this dataset.",
    icon="\u2139\ufe0f",
)

df, cluster_profiles = load_and_score_data()

st.subheader("At a glance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total customers", f"{len(df):,}")
c2.metric("Churn rate", f"{(df['Churn'] == 'Yes').mean():.1%}")
c3.metric("High churn risk", f"{(df['Churn_Risk'] == 'High').sum():,}")
c4.metric("Avg predicted 12mo LTV", f"${df['predicted_ltv'].mean():,.0f}")

st.subheader("Priority distribution")
st.caption(
    "Every customer falls into exactly one of 6 cells (Churn Risk x LTV "
    "Tier) -- this is the canonical, algorithmic recommendation logic "
    "(see the Recommendations page for the interactive table)."
)
priority_counts = (
    df["Priority_Label"].value_counts().rename_axis("Priority").reset_index(name="Customers")
)
st.dataframe(priority_counts, width='stretch', hide_index=True)

st.markdown("---")
st.markdown(
    "Use the sidebar to navigate: **Executive Overview**, **Segments**, "
    "**Churn Prediction**, **LTV Prediction**, **SHAP Explainability**, "
    "and **Recommendations**."
)

st.caption(
    "Predicted churn probabilities are not calibrated (scale_pos_weight "
    "skews them for recall); the 0.5 threshold is used for ranking/"
    "prioritization, not as a literal probability."
)