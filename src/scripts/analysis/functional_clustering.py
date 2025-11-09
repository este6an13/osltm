"""
Functional Clustering of Stations by Day Type
=============================================

This script computes the mean normalized λ(t) curve per station × day type (WD, SA, SU, HO)
for all stations in the database, and clusters the stations based on their functional shape.

It uses KMeans for main clustering, and optionally hierarchical clustering for interpretation.
"""

from itertools import combinations
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Import your project dependencies
from src.db.session_v1 import SessionLocal
from src.repo.v1.estimates.models import Estimate
from src.repo.v1.stations.models import Station

# ==========================================================
# 0. Load and prepare data
# ==========================================================


def _load_estimate_data() -> pd.DataFrame:
    """Load all estimate records from the database."""
    from sqlalchemy.orm import Session

    session: Session = SessionLocal()
    query = session.query(Estimate)
    df = pd.read_sql(query.statement, session.bind)
    session.close()
    return df


def _load_station_info() -> pd.DataFrame:
    """Load station id, code, and name from DB."""
    from sqlalchemy.orm import Session

    session: Session = SessionLocal()
    query = session.query(Station.id, Station.code, Station.name)
    df = pd.read_sql(query.statement, session.bind)
    session.close()
    return df


def _reshape_to_daily_functions(df: pd.DataFrame, lambda_col: str) -> pd.DataFrame:
    """Convert raw 15-min records into one vector per day (discrete function)."""
    grouped = (
        df.groupby(["station_id", "day", "date_type", "month", "day_of_week"])[
            lambda_col
        ]
        .apply(list)
        .reset_index()
        .rename(columns={lambda_col: "lambda_curve"})
    )
    grouped["n_points"] = grouped["lambda_curve"].apply(len)
    grouped = grouped[grouped["n_points"] >= 80]  # tolerate some missing windows
    return grouped


def _align_and_interpolate_curves(
    curves: list[np.ndarray], target_len: int = 96
) -> np.ndarray:
    """Interpolate each curve to same length and fill NaNs."""
    aligned = []
    for arr in curves:
        arr = np.array(arr, dtype=float)
        if np.sum(~np.isnan(arr)) < 10:
            continue
        x = np.arange(len(arr))
        if np.any(np.isnan(arr)):
            valid = ~np.isnan(arr)
            arr = np.interp(x, x[valid], arr[valid])
        # Interpolate to common length
        n = len(arr)
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_len)
        arr_interp = np.interp(x_new, x_old, arr)
        aligned.append(arr_interp)
    if not aligned:
        return np.empty((0, target_len))
    return np.stack(aligned)


def test_clustering_stability_and_quality(
    feature_df: pd.DataFrame,
    k_values: list[int] = [2, 3, 4, 5, 6],
    seeds: list[int] = [0, 42, 123],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Test clustering stability (ARI) and quality (Silhouette) for several k values.
    Returns a summary DataFrame with mean ARI and mean Silhouette for each k.
    """
    # Prepare data
    X = feature_df[[c for c in feature_df.columns if c.startswith("t_")]].values
    if normalize:
        X = StandardScaler().fit_transform(X)

    results = []

    for k in k_values:
        labels_by_seed = {}

        # Fit KMeans with multiple seeds
        for seed in seeds:
            km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            labels_by_seed[seed] = km.fit_predict(X)

        # Compute pairwise ARIs between seeds
        ari_scores = []
        for s1, s2 in combinations(seeds, 2):
            ari = adjusted_rand_score(labels_by_seed[s1], labels_by_seed[s2])
            ari_scores.append(ari)

        mean_ari = np.mean(ari_scores)

        # Compute silhouette score for one representative seed (e.g. first)
        silhouette = silhouette_score(X, labels_by_seed[seeds[0]])

        results.append(
            {
                "n_clusters": k,
                "mean_ARI": mean_ari,
                "silhouette": silhouette,
            }
        )

    return pd.DataFrame(results)


# ==========================================================
# 1. Compute mean functional curve per station × date_type
# ==========================================================


def get_station_daytype_signatures(
    lambda_type: Literal["in", "out"] = "in",
    normalize: bool = True,
) -> pd.DataFrame:
    """Return one 96-dim vector per station × date_type (WD/SA/SU/HO)."""
    df = _load_estimate_data()
    value_col = "estimated_lambda_in" if lambda_type == "in" else "estimated_lambda_out"
    daily = _reshape_to_daily_functions(df, value_col)

    features = []
    for (station_id, date_type), g in daily.groupby(["station_id", "date_type"]):
        curves = []
        for row in g.itertuples():
            arr = np.array(row.lambda_curve, dtype=float)
            if normalize:
                total = np.nansum(arr)
                if total > 0:
                    arr = arr / total
            curves.append(arr)
        if len(curves) == 0:
            continue
        curves = _align_and_interpolate_curves(curves)
        if curves.size == 0:
            continue
        mean_curve = np.nanmean(curves, axis=0)
        features.append(
            {"station_id": station_id, "date_type": date_type, "curve": mean_curve}
        )

    return pd.DataFrame(features)


def flatten_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the curve column into separate feature columns (t_00 ... t_95)."""
    expanded = pd.DataFrame(df["curve"].to_list())
    expanded.columns = [f"t_{i:02d}" for i in range(expanded.shape[1])]
    expanded["station_id"] = df["station_id"]
    expanded["date_type"] = df["date_type"]
    return expanded


# ==========================================================
# 2. KMeans clustering per day type
# ==========================================================


def cluster_stations_by_daytype(
    features_df: pd.DataFrame,
    n_clusters_by_type: dict[str, int],
    normalize: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster stations based on mean functional pattern for each date_type.
    Allows specifying a custom number of clusters per date_type.

    Args:
        features_df: DataFrame with station_id, date_type, and curve features
        n_clusters_by_type: dict, e.g. {"WD": 3, "SA": 2, "SU": 2, "HO": 3}
        normalize: scale features to zero mean, unit variance
        seed: random seed for reproducibility

    Returns:
        cluster_df: full detailed dataframe (with all curve features + cluster)
        summary: compact dataframe (station_id, code, name, date_type, cluster)
    """
    feature_mat = flatten_feature_matrix(features_df)
    results = []

    for date_type, g in feature_mat.groupby("date_type"):
        n_clusters = n_clusters_by_type.get(date_type, 3)
        X = g[[c for c in g.columns if c.startswith("t_")]].values

        if X.shape[0] < n_clusters:
            print(
                f"⚠️ Skipping {date_type} — only {X.shape[0]} stations (< {n_clusters})"
            )
            continue

        if normalize:
            X = StandardScaler().fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)

        g = g.copy()
        g["cluster"] = labels
        results.append(g)

        # Plot mean curve per cluster
        plt.figure(figsize=(8, 4))
        for c in range(n_clusters):
            mean_curve = np.mean(X[labels == c], axis=0)
            plt.plot(mean_curve, label=f"Cluster {c} (n={np.sum(labels == c)})")
        plt.title(f"Station Clusters for {date_type} (k={n_clusters})")
        plt.xlabel("15-min interval")
        plt.ylabel("Normalized λ(t)")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    cluster_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # Merge with station info
    station_info = _load_station_info()
    cluster_df = cluster_df.merge(
        station_info, left_on="station_id", right_on="id", how="left"
    )

    # Compact summary
    summary = cluster_df[
        ["station_id", "code", "name", "date_type", "cluster"]
    ].sort_values(["date_type", "cluster", "station_id"])

    return cluster_df, summary


def cluster_stations_by_daytype_gmm(
    features_df: pd.DataFrame,
    n_clusters_by_type: dict[str, int],
    normalize: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster stations based on mean functional pattern for each date_type
    using Gaussian Mixture Models (GMM).

    Allows specifying a custom number of clusters per date_type.

    Args:
        features_df: DataFrame with station_id, date_type, and curve features
        n_clusters_by_type: dict, e.g. {"WD": 3, "SA": 2, "SU": 2, "HO": 3}
        normalize: scale features to zero mean, unit variance
        seed: random seed for reproducibility

    Returns:
        cluster_df: full detailed dataframe (with all curve features + cluster)
        summary: compact dataframe (station_id, code, name, date_type, cluster)
    """
    feature_mat = flatten_feature_matrix(features_df)
    results = []

    for date_type, g in feature_mat.groupby("date_type"):
        n_clusters = n_clusters_by_type.get(date_type, 3)
        X = g[[c for c in g.columns if c.startswith("t_")]].values

        if X.shape[0] < n_clusters:
            print(
                f"⚠️ Skipping {date_type} — only {X.shape[0]} stations (< {n_clusters})"
            )
            continue

        if normalize:
            X = StandardScaler().fit_transform(X)

        # --- GMM clustering ---
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",  # can also try "tied" or "diag"
            random_state=seed,
        )
        labels = gmm.fit_predict(X)

        g = g.copy()
        g["cluster"] = labels
        results.append(g)

        # --- Visualization: mean curves per cluster ---
        plt.figure(figsize=(8, 4))
        for c in range(n_clusters):
            mean_curve = np.mean(X[labels == c], axis=0)
            plt.plot(mean_curve, label=f"Cluster {c} (n={np.sum(labels == c)})")
        plt.title(f"GMM Station Clusters for {date_type} (k={n_clusters})")
        plt.xlabel("15-min interval")
        plt.ylabel("Normalized λ(t)")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    # --- Merge results ---
    cluster_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # Merge with station info
    station_info = _load_station_info()
    cluster_df = cluster_df.merge(
        station_info, left_on="station_id", right_on="id", how="left"
    )

    # Compact summary
    summary = cluster_df[
        ["station_id", "code", "name", "date_type", "cluster"]
    ].sort_values(["date_type", "cluster", "station_id"])

    return cluster_df, summary


# ==========================================================
# 3. Hierarchical clustering
# ==========================================================


def hierarchical_station_clustering(
    features_df: pd.DataFrame, date_type: str = "WD", metric: str = "correlation"
):
    """Plot hierarchical clustering dendrogram for one day type."""
    g = flatten_feature_matrix(features_df)
    g = g[g["date_type"] == date_type]
    X = g[[c for c in g.columns if c.startswith("t_")]].values
    dist = pairwise_distances(X, metric=metric)
    Z = linkage(dist, method="average")
    plt.figure(figsize=(10, 4))
    dendrogram(Z, labels=g["station_id"].values)
    plt.title(f"Hierarchical Clustering — {date_type}")
    plt.tight_layout()
    plt.show()
    labels = fcluster(Z, t=4, criterion="maxclust")
    g["cluster"] = labels
    return g


# ==========================================================
# 4. PCA visualization
# ==========================================================


def visualize_clusters_per_date_type(cluster_df: pd.DataFrame):
    """
    2D PCA visualization of clustered station curves, plotted separately per date_type.
    Each subplot shows the PCA projection of stations for that day type, colored by cluster.
    """
    feature_cols = [c for c in cluster_df.columns if c.startswith("t_")]
    date_types = sorted(cluster_df["date_type"].unique())

    for dt in date_types:
        g = cluster_df[cluster_df["date_type"] == dt].copy()
        if g.empty:
            continue

        # Perform PCA on curves for this date_type
        X = g[feature_cols].values
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X)
        g["PC1"], g["PC2"] = pcs[:, 0], pcs[:, 1]

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=g,
            x="PC1",
            y="PC2",
            hue="cluster",
            palette="tab10",
            s=80,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.7,
        )

        plt.title(f"PCA of Station Clusters — {dt}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
