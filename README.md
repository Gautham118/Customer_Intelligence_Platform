# Customer Intelligence Platform

**Segmentation, Churn Prediction, LTV Modeling & Retention Recommendations — built end-to-end on the Telco Customer Churn dataset.**

> Can we identify high-value customers who are likely to churn, and recommend the most profitable retention strategy for each customer segment?

This project is a full data science pipeline — from raw data to an interactive Streamlit dashboard — that answers that question. It combines RFM-based customer segmentation, an XGBoost churn classifier, an XGBoost LTV regressor, SHAP explainability, and a rules-based recommendation engine into one coherent business narrative: **segment → churn risk → LTV → recommended action.**

---

## Table of Contents

- [Overview](#overview)
- [Dataset & Synthetic Feature Disclosure](#dataset--synthetic-feature-disclosure)
- [Methodology](#methodology)
  - [1. Data Cleaning](#1-data-cleaning)
  - [2. Feature Engineering — RFM & Behavioral Features](#2-feature-engineering--rfm--behavioral-features)
  - [3. Customer Segmentation](#3-customer-segmentation)
  - [4. Churn Prediction](#4-churn-prediction)
  - [5. LTV Prediction](#5-ltv-prediction)
  - [6. Explainability (SHAP)](#6-explainability-shap)
  - [7. Churn Risk × LTV Priority Matrix](#7-churn-risk--ltv-priority-matrix)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Example Insights](#example-insights)
- [Known Limitations & Caveats](#known-limitations--caveats)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Telecom subscription businesses lose significant revenue to churn, but not every churning customer is equally worth saving. This project builds a pipeline that:

1. **Segments** customers using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering
2. **Predicts churn risk** with an XGBoost classifier, tuned for recall on the minority (churn) class
3. **Predicts 12-month customer lifetime value (LTV)** with an XGBoost regressor
4. **Explains predictions** with SHAP, at both the global and individual-customer level
5. **Recommends an action per customer** using a Churn Risk × LTV priority matrix, so retention spend goes where it has the most business impact
6. Ties it all together in a **6-page interactive Streamlit dashboard**

The project prioritizes a correct, leakage-free, end-to-end pipeline and a clear business narrative over speed — every modeling decision below is deliberate and documented so it can hold up under interview-level scrutiny.

---

## Dataset & Synthetic Feature Disclosure

- **Source**: [Telco Customer Churn dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 original columns.
- **Data quirk handled**: 11 rows have a blank `TotalCharges` value, all belonging to brand-new customers (`tenure == 0`). These are coerced to numeric and imputed to `0.0` rather than dropped, since dropping would discard otherwise-complete records over a single explainable field.

**⚠️ Important disclosure:** The raw Telco dataset has no transaction log or forward-looking revenue field, so it can't natively support RFM analysis or LTV modeling. To build those components, this project generates **synthetic-but-realistically-distributed behavioral features** (e.g. session duration, days since last activity, spend trend) and a **synthetic LTV target** (`future_ltv_12m`). This is a deliberate, disclosed design choice made to simulate what a richer real-world dataset would contain — not an attempt to pass off simulated data as real.

- Every synthetic column is generated as a function of real columns (`tenure`, `MonthlyCharges`, `Contract`, `InternetService`) plus fixed-seed randomness — **never** as a function of the `Churn` label itself, to avoid leaking the target into feature generation.
- `Churn` is used *after* generation only, to sanity-check that the resulting distributions are directionally plausible (e.g. do churners skew toward lower recent activity?).
- The LTV target `future_ltv_12m` is explicitly a **disclosed simulated proxy**, not observed revenue — see [LTV Prediction](#5-ltv-prediction) below for the exact formula.

---

## Methodology

### 1. Data Cleaning
`src/preprocessing.py` coerces `TotalCharges` to numeric and imputes the 11 blank/whitespace rows (all `tenure == 0`) to `0.0`.

### 2. Feature Engineering — RFM & Behavioral Features
`src/feature_engineering.py` generates 7 synthetic behavioral columns (e.g. `avg_session_duration_min`, `days_since_last_activity`, `days_active_last_90d`, `monthly_spend_trend_pct`) and derives RFM scores from them:

| RFM Component | Derived From | Notes |
|---|---|---|
| **Recency** | `days_since_last_activity` (inverted) | Fewer days since last activity → higher score |
| **Frequency** | `days_active_last_90d` | More active days in the window → higher score |
| **Monetary** | `MonthlyCharges` | Real (non-synthetic) column |

Each component is scored on a 1–5 scale using quintiles (`pd.qcut`, with a percentile-rank fallback for low-variance columns), then combined into an `RFM_Score` (e.g. `"543"`) and an `RFM_Segment` label.

### 3. Customer Segmentation
`src/segmentation.py` runs **K-Means clustering** on the RFM scores. The number of clusters (`k`) is chosen using both the **Elbow Method** and **Silhouette Score** (tested across k = 3–8), not hardcoded. Resulting clusters are mapped to a fixed business-friendly taxonomy — **Champions, Loyal Customers, At Risk, New/Low Engagement, Lost** — based on which archetype each cluster's RFM profile is closest to. A cluster that doesn't clearly match any archetype is flagged rather than force-labeled.

### 4. Churn Prediction
`src/churn_model.py` trains an **XGBoost classifier** (`enable_categorical=True`, `tree_method="hist"`) to predict churn probability.

- **Class imbalance** (~26.5% positive class) handled via `scale_pos_weight`, recomputed from each training fold's own rows — never from the full dataset — to avoid leakage.
- **Evaluation**: Stratified 5-fold CV, reporting **Recall, ROC-AUC, and PR-AUC** — not raw accuracy, since a missed churner costs more than a wasted retention offer, and PR-AUC is the more honest metric under class imbalance.
- **Calibration caveat**: `scale_pos_weight` improves minority-class recall but skews predicted probabilities away from true calibrated probabilities. The 0.5 threshold is used for **ranking/prioritization**, not as a literal probability.
- Achieves ~0.73 recall, ~0.83 ROC-AUC, ~0.63 PR-AUC on cross-validation.

### 5. LTV Prediction
`src/ltv_model.py` trains an **XGBoost regressor** to predict `future_ltv_12m`, a disclosed synthetic proxy target:

```
future_ltv_12m = (MonthlyCharges × 12) × retention_multiplier + noise
```

- `retention_multiplier` is much lower for churned customers (truncated future spend) and near/above 1.0 for retained customers with stable or growing spend trends.
- Gaussian noise (~7.5% of base value) ensures the target isn't a deterministic function of its own inputs.
- **Leakage guardrails**: the ground-truth `Churn` label is excluded from the regressor's feature set — only the churn classifier's **out-of-fold predicted probability** is allowed in as a churn-signal feature. `TotalCharges` is also excluded (collinear with `tenure × MonthlyCharges`).
- **Evaluation**: MAE, RMSE, R² (MAPE as an extra). Test R² ≈ 0.59 — deliberately well below the ~0.99 a naive `tenure × MonthlyCharges` reconstruction would produce, confirming the model is learning from segment/behavioral drivers rather than reconstructing its own target formula.
- Predicted LTV is bucketed into three tiers: **High (top 25%)**, **Medium (50th–74th percentile)**, **Low (bottom 50%)**.

### 6. Explainability (SHAP)
`src/explainability.py` uses `shap.TreeExplainer` (works for both the churn classifier and LTV regressor, since both are XGBoost tree models) to generate:
- **Global** feature importance (mean absolute SHAP value across all customers)
- **Local** per-customer waterfall explanations, translated into plain-English narratives (e.g. *"This customer has a predicted churn probability of 83%. Factors increasing churn risk: Contract=Month-to-month, InternetService=Fiber optic..."*)

### 7. Churn Risk × LTV Priority Matrix
`src/recommendations.py` implements the canonical, algorithmic per-customer recommendation logic:

| Churn Risk | LTV | Priority Label | Action |
|---|---|---|---|
| High | High | **Save Immediately** | Personalized retention offer + priority outreach |
| High | Medium | **Targeted Retention** | Lighter-touch retention offer |
| High | Low | **Low Priority** | Minimal/no intervention |
| Low | High | **Maintain Loyalty** | Loyalty rewards, upsell premium features |
| Low | Medium | **Grow Engagement** | Standard engagement + light upsell nudges |
| Low | Low | **Monitor** | No action needed |

A separate Segment → Action table exists as a narrative overlay for segment-level dashboard messaging, but the priority matrix above is what actually drives every per-customer recommendation — if the two ever disagree for a specific customer, the priority matrix wins.

---

## Dashboard

A 6-page **Streamlit** dashboard ties the whole pipeline together:

| Page | Contents |
|---|---|
| 1. Executive Overview | KPIs — churn rate, avg LTV, high-risk customer count, segment breakdown |
| 2. Segments | Cluster profiles, RFM distributions, segment naming rationale |
| 3. Churn Prediction | Model performance, risk distribution, high-risk customer table |
| 4. LTV Prediction | Predicted LTV distribution, tier breakdown, model performance |
| 5. SHAP Explainability | Global feature importance + individual customer narratives |
| 6. Recommendations | Interactive Churn Risk × LTV priority matrix table, filterable by segment/tier |

> **Note**: v1 prioritizes full functional correctness across all 6 pages over visual styling — visual polish is a planned follow-up pass.

---

## Project Structure

```
customer-intelligence-platform/
├── data/
│   ├── raw/                 # original Telco CSV (gitignored)
│   ├── interim/              # cleaned data
│   └── processed/            # feature-engineered dataset(s)
├── notebooks/                # EDA + prototyping (01–06)
├── src/
│   ├── config.py              # thresholds, paths, constants
│   ├── data_loader.py         # load + validate raw CSV
│   ├── preprocessing.py       # cleaning, dtype fixes
│   ├── feature_engineering.py # RFM + synthetic behavioral features + LTV target
│   ├── segmentation.py        # K-Means clustering + cluster naming
│   ├── churn_model.py         # XGBoost churn classifier
│   ├── ltv_model.py           # XGBoost LTV regressor
│   ├── explainability.py      # SHAP global + local explanations
│   ├── recommendations.py     # Churn Risk × LTV priority matrix
│   └── utils.py                # shared helpers
├── models/                    # saved .joblib model artifacts (gitignored)
├── app/
│   ├── streamlit_app.py       # dashboard entrypoint
│   └── pages/                 # 6-page multipage app
├── tests/                      # pytest unit tests
├── environment.yml
├── pyproject.toml
└── README.md
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Gautham118/Customer_Intelligence_Platform.git
cd Customer_Intelligence_Platform

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate ml_env

# 3. Place the raw dataset
# Download "Telco Customer Churn" from Kaggle and place it at:
# data/raw/telco_customer_churn.csv

# 4. Run tests
pytest

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

The package is installed in editable mode (`pip install -e .`, via `environment.yml`) so `src/` modules are importable from notebooks and the Streamlit app.

---

## Example Insights

- **Contract type and internet service type are the strongest churn drivers** — month-to-month contracts and fiber-optic customers show materially higher churn risk than longer-term contracts or DSL/no-internet customers.
- Roughly **36% of the customer base** is flagged as high churn risk (probability ≥ 0.5), but only a subset of those are also high-LTV — the priority matrix filters this down to the customers where retention spend is actually worth it.
- The LTV model's top predictive signals are RFM segment and cluster assignment, reinforcing that behavioral/engagement patterns — not just raw billing amount — drive projected future value.

---

## Known Limitations & Caveats

- **Synthetic data**: behavioral features and the LTV target are simulated, disclosed proxies — not observed real-world transactional data. See [Dataset & Synthetic Feature Disclosure](#dataset--synthetic-feature-disclosure).
- **Uncalibrated churn probabilities**: `scale_pos_weight` trades probability calibration for better recall on the minority class. Probabilities should be read as a ranking signal, not literal likelihoods.
- **Out of scope for v1** (by design): survival analysis, BG/NBD or Gamma-Gamma probabilistic LTV models, uplift/causal modeling, database persistence, cloud deployment. Dockerization is a stretch goal for after v1 polish.

---

## Tech Stack

Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · SHAP · Streamlit · Plotly · pytest · black · ruff

---

## License

MIT — see [LICENSE](LICENSE).
