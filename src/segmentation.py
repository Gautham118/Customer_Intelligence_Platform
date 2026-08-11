"""K-Means customer segmentation: k selection support + cluster naming.

Depends on RFM scores from feature_engineering.py (CLAUDE.md Section 9
dependency order): run engineer_features() before calling anything here.

k is NOT hardcoded (CLAUDE.md Section 6): this module exposes elbow/
silhouette metrics across a k range for visual review in
03_segmentation.ipynb, plus a silhouette-based *suggestion* as a
convenience default. Notebooks are for prototyping/narrative (Section 5) --
the mechanical logic lives here, but the final k choice and reasoning
should be documented in the notebook.

Clustering feature set: the three RFM quintile scores (R_score, F_score,
M_score), NOT the full behavioral feature set. This matches CLAUDE.md
Section 6's taxonomy, which is defined entirely in R/F/M terms ("high R,
high F, high M" etc.) -- clustering directly on those three scores keeps
the resulting clusters interpretable against that taxonomy. No
standardization is applied since all three already share the same native
1-5 scale by construction (pd.qcut quintiles).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import KMEANS_K_MAX, KMEANS_K_MIN, RANDOM_STATE

logger = logging.getLogger(__name__)

CLUSTERING_FEATURES = ["R_score", "F_score", "M_score"]

# Reference (R, F, M) points for each taxonomy label (CLAUDE.md Section 6).
# All on the native 1-5 RFM score scale.
TAXONOMY_ARCHETYPES = {
    "Champions": (5, 5, 5),            # high R, high F, high M
    "Loyal Customers": (4, 5, 3),      # high F, moderate M, moderate-high R
    "At Risk": (2, 4, 4),              # declining R, previously high F/M
    "New/Low Engagement": (4, 1, 2),   # recent signup (high R) but low F, low tenure
    "Lost": (1, 1, 1),                 # low R, low F, low M
}

# A cluster whose nearest archetype is farther than this (Euclidean, raw
# R/F/M score units) doesn't "clearly match" any taxonomy label per
# CLAUDE.md Section 6 -- it gets flagged instead of forced. This is my own
# default, not a CLAUDE.md-locked number; worth revisiting once real
# cluster profiles are visible.
AMBIGUOUS_MATCH_THRESHOLD = 2.0


def find_optimal_k(
    df: pd.DataFrame,
    k_min: int = KMEANS_K_MIN,
    k_max: int = KMEANS_K_MAX,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute elbow (inertia) and silhouette metrics across a range of k.

    Does NOT pick k -- CLAUDE.md Section 6 requires showing both curves
    before deciding, typically in 03_segmentation.ipynb. Use `suggest_k` on
    the result for a starting-point default.

    Args:
        df: Dataframe with RFM score columns (post `add_rfm_scores`).
        k_min: Smallest k to test (inclusive).
        k_max: Largest k to test (inclusive).
        random_state: KMeans random state, for reproducibility.

    Returns:
        Dataframe with one row per k: ``k``, ``inertia`` (elbow metric),
        ``silhouette`` (higher is better, range [-1, 1]).
    """
    X = df[CLUSTERING_FEATURES].to_numpy()
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        logger.info("k=%d: inertia=%.1f, silhouette=%.4f", k, km.inertia_, sil)
    return pd.DataFrame(rows)


def suggest_k(k_metrics: pd.DataFrame) -> int:
    """Suggest a k as the highest-silhouette candidate.

    A convenience default only -- cross-check against the elbow curve
    before treating this as final (CLAUDE.md Section 6 requires both).

    Args:
        k_metrics: Output of `find_optimal_k`.

    Returns:
        The k value with the highest silhouette score.
    """
    best_row = k_metrics.loc[k_metrics["silhouette"].idxmax()]
    return int(best_row["k"])


def fit_kmeans(df: pd.DataFrame, k: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Fit K-Means on the RFM score features and return cluster labels.

    Args:
        df: Dataframe with RFM score columns.
        k: Number of clusters.
        random_state: KMeans random state, for reproducibility.

    Returns:
        Array of cluster labels (0..k-1), one per row of ``df``.
    """
    X = df[CLUSTERING_FEATURES].to_numpy()
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    logger.info("Fit K-Means with k=%d, final inertia=%.1f", k, km.inertia_)
    return labels


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each cluster's RFM profile plus business-context stats.

    Args:
        df: Dataframe with ``Cluster_ID`` and RFM score columns. ``tenure``,
            ``Churn``, and ``future_ltv_12m`` are included in the summary
            when present, for narrative context beyond raw R/F/M.

    Returns:
        Dataframe indexed by ``Cluster_ID`` with mean R/F/M scores, cluster
        size, and (if available) mean tenure, churn rate, and mean LTV.
    """
    agg = {
        "R_score": "mean", "F_score": "mean", "M_score": "mean",
        "customerID": "count",
    }
    if "tenure" in df.columns:
        agg["tenure"] = "mean"
    if "future_ltv_12m" in df.columns:
        agg["future_ltv_12m"] = "mean"

    profiles = df.groupby("Cluster_ID").agg(agg).rename(columns={"customerID": "size"})

    if "Churn" in df.columns:
        profiles["churn_rate"] = df.groupby("Cluster_ID")["Churn"].apply(
            lambda s: (s == "Yes").mean()
        )
    return profiles.round(2)


def map_clusters_to_taxonomy(profiles: pd.DataFrame) -> pd.DataFrame:
    """Map each cluster to its nearest taxonomy label (or flag it).

    For each cluster, finds the nearest of the 5 taxonomy archetypes
    (Euclidean distance in raw R/F/M score space). Multiple clusters can
    share a label (CLAUDE.md Section 6 explicitly allows this). If the
    nearest archetype is farther than `AMBIGUOUS_MATCH_THRESHOLD`, the
    cluster is flagged as unmapped rather than forced into a label.

    Args:
        profiles: Output of `build_cluster_profiles` (must have
            R_score/F_score/M_score mean columns).

    Returns:
        Dataframe indexed by ``Cluster_ID`` with ``Segment_Name``,
        ``nearest_distance``, and ``ambiguous`` (bool) columns.
    """
    archetype_names = list(TAXONOMY_ARCHETYPES.keys())
    archetype_points = np.array(list(TAXONOMY_ARCHETYPES.values()))

    results = []
    for cluster_id, row in profiles.iterrows():
        point = np.array([row["R_score"], row["F_score"], row["M_score"]])
        distances = np.linalg.norm(archetype_points - point, axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_idx])
        ambiguous = nearest_dist > AMBIGUOUS_MATCH_THRESHOLD

        if ambiguous:
            label = f"Unmapped (Cluster {cluster_id})"
            logger.warning(
                "Cluster %s (R=%.2f, F=%.2f, M=%.2f) is %.2f from its nearest "
                "archetype '%s' -- beyond the %.1f match threshold. Doesn't "
                "clearly match any taxonomy label; review manually rather "
                "than trusting this mapping.",
                cluster_id, row["R_score"], row["F_score"], row["M_score"],
                nearest_dist, archetype_names[nearest_idx], AMBIGUOUS_MATCH_THRESHOLD,
            )
        else:
            label = archetype_names[nearest_idx]

        results.append({
            "Cluster_ID": cluster_id,
            "Segment_Name": label,
            "nearest_distance": round(nearest_dist, 3),
            "ambiguous": ambiguous,
        })

    return pd.DataFrame(results).set_index("Cluster_ID")


def segment_customers(
    df: pd.DataFrame,
    k: Optional[int] = None,
    k_min: int = KMEANS_K_MIN,
    k_max: int = KMEANS_K_MAX,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full segmentation pipeline: fit K-Means, name the clusters.

    Args:
        df: Dataframe with RFM scores (post `add_rfm_scores`).
        k: Number of clusters. If None, runs `find_optimal_k` across
            [k_min, k_max] and auto-picks the highest-silhouette k as a
            provisional default -- logs a warning that this should be
            cross-checked against the elbow/silhouette plot in
            03_segmentation.ipynb before being treated as final.
        k_min: Smallest k to search, if `k` is None.
        k_max: Largest k to search, if `k` is None.
        random_state: KMeans random state, for reproducibility.

    Returns:
        Tuple of:
          - ``df`` copy with ``Cluster_ID`` and ``Segment_Name`` columns added.
          - Cluster profile summary dataframe (from `build_cluster_profiles`,
            joined with the taxonomy mapping) for inspection/notebook display.
    """
    df = df.copy()

    if k is None:
        k_metrics = find_optimal_k(df, k_min, k_max, random_state)
        k = suggest_k(k_metrics)
        logger.warning(
            "No k specified -- auto-selected k=%d based on highest silhouette "
            "score. This is a provisional default; confirm against the "
            "elbow/silhouette plot in 03_segmentation.ipynb before finalizing.",
            k,
        )

    df["Cluster_ID"] = fit_kmeans(df, k, random_state)
    profiles = build_cluster_profiles(df)
    mapping = map_clusters_to_taxonomy(profiles)

    df["Segment_Name"] = df["Cluster_ID"].map(mapping["Segment_Name"])
    profiles = profiles.join(mapping)

    logger.info("Segmentation complete (k=%d):\n%s", k, profiles.to_string())
    return df, profiles