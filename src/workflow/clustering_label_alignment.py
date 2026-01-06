"""
Clustering + Label Alignment (Exploratory Sanity Check)

This script clusters daily count profiles and compares cluster labels to day-type labels
to assess whether clusters align with the day-type taxonomy (WD, SA, SU, HO).

Questions addressed:
- Do clusters roughly match the day-type taxonomy?
- Are holidays more heterogeneous?
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.workflow.data_loader import load_data

# Color mapping for date types
DATE_TYPE_COLORS = {
    "WD": "#2E86AB",  # Blue
    "SA": "#A23B72",  # Purple
    "SU": "#F18F01",  # Orange
    "HO": "#C73E1D",  # Red
}

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

# Cluster colors (distinct from date type colors)
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def sort_time_columns(time_cols: list[str]) -> list[str]:
    """
    Sort time column names by their numeric time value.

    Args:
        time_cols: List of time column names (e.g., ["t_400", "t_415", "t_1000"])

    Returns:
        Sorted list by numeric time value
    """

    def get_time_value(col: str) -> int:
        time_str = col.replace("t_", "")
        return int(time_str)

    return sorted(time_cols, key=get_time_value)


def extract_daily_profiles(
    df: pd.DataFrame,
    station_code: Optional[str] = None,
) -> tuple[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Extract daily profiles (time series vectors) from the DataFrame.

    Args:
        df: DataFrame from data_loader with time window columns
        station_code: Optional station code to filter by. If None, uses all stations.

    Returns:
        Tuple of (profiles_matrix, date_types, metadata_df) where:
        - profiles_matrix: (n_samples, n_time_windows) array of daily profiles
        - date_types: Series of date type labels (WD, SA, SU, HO)
        - metadata_df: DataFrame with year, month, day, station_code, station_name
    """
    # Filter by station if specified
    if station_code is not None:
        df = df[df["station_code"] == station_code].copy()

    if df.empty:
        return np.array([]), pd.Series(dtype=str), pd.DataFrame()

    # Get time columns and sort them numerically
    time_cols = [c for c in df.columns if c.startswith("t_")]
    time_cols = sort_time_columns(time_cols)

    if not time_cols:
        return np.array([]), pd.Series(dtype=str), pd.DataFrame()

    # Extract profiles (each row is a daily profile)
    profiles = df[time_cols].values

    # Fill NaN with 0 (or could use forward fill, but 0 is reasonable for counts)
    profiles = np.nan_to_num(profiles, nan=0.0)

    # Extract metadata
    date_types = df["date_type"].values
    metadata = df[["year", "month", "day", "station_code", "station_name"]].copy()

    return profiles, pd.Series(date_types), metadata


def cluster_profiles(
    profiles: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    normalize: bool = True,
) -> tuple[np.ndarray, KMeans]:
    """
    Cluster daily profiles using k-means.

    Args:
        profiles: (n_samples, n_features) array of daily profiles
        n_clusters: Number of clusters (default: 3)
        random_state: Random seed for reproducibility
        normalize: Whether to normalize profiles before clustering

    Returns:
        Tuple of (cluster_labels, kmeans_model)
    """
    if len(profiles) == 0:
        return np.array([]), None

    # Normalize if requested (standardize each time window)
    if normalize:
        scaler = StandardScaler()
        profiles_scaled = scaler.fit_transform(profiles)
    else:
        profiles_scaled = profiles

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(profiles_scaled)

    return cluster_labels, kmeans


def compute_alignment_metrics(
    cluster_labels: np.ndarray,
    date_types: pd.Series,
) -> dict:
    """
    Compute metrics comparing cluster labels to date type labels.

    Args:
        cluster_labels: Array of cluster assignments
        date_types: Series of date type labels (WD, SA, SU, HO)

    Returns:
        Dictionary with alignment metrics
    """
    if len(cluster_labels) == 0:
        return {}

    # Convert date types to numeric for ARI calculation
    date_type_map = {"WD": 0, "SA": 1, "SU": 2, "HO": 3}
    date_type_numeric = date_types.map(date_type_map).values

    # Adjusted Rand Index (measures agreement between two clusterings)
    ari = adjusted_rand_score(date_type_numeric, cluster_labels)

    # Confusion matrix
    cm = confusion_matrix(date_type_numeric, cluster_labels)

    # Compute purity for each cluster (most common date type in each cluster)
    cluster_purities = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        cluster_date_types = date_types[mask]
        if len(cluster_date_types) > 0:
            most_common = cluster_date_types.mode()[0]
            count = (cluster_date_types == most_common).sum()
            purity = count / len(cluster_date_types)
            cluster_purities[int(cluster_id)] = {
                "most_common_type": most_common,
                "purity": purity,
                "size": len(cluster_date_types),
            }

    # Compute heterogeneity for each date type (spread across clusters)
    date_type_heterogeneity = {}
    for date_type in date_types.unique():
        mask = date_types == date_type
        type_clusters = cluster_labels[mask]
        unique_clusters = len(np.unique(type_clusters))
        total_count = len(type_clusters)

        # Compute entropy (higher = more spread)
        cluster_counts = pd.Series(type_clusters).value_counts()
        proportions = cluster_counts / total_count
        entropy = -(proportions * np.log(proportions + 1e-10)).sum()

        date_type_heterogeneity[date_type] = {
            "n_clusters": unique_clusters,
            "entropy": entropy,
            "total_count": total_count,
        }

    return {
        "adjusted_rand_index": ari,
        "confusion_matrix": cm,
        "cluster_purities": cluster_purities,
        "date_type_heterogeneity": date_type_heterogeneity,
    }


def plot_clustering_results(
    profiles: np.ndarray,
    cluster_labels: np.ndarray,
    date_types: pd.Series,
    metrics: dict,
    station_code: Optional[str] = None,
    station_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    count_type: Literal["in", "out"] = "in",
    n_clusters: int = 3,
):
    """
    Create visualization of clustering results.

    Args:
        profiles: (n_samples, n_features) array of daily profiles
        cluster_labels: Array of cluster assignments
        date_types: Series of date type labels
        metrics: Dictionary with alignment metrics
        station_code: Station code (for title/filename)
        station_name: Station name (for title)
        output_dir: Output directory for saving plots
        count_type: Type of counts ("in" or "out")
        n_clusters: Number of clusters
    """
    if len(profiles) == 0:
        print("⚠️  No profiles to plot")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. PCA projection colored by cluster
    ax1 = fig.add_subplot(gs[0, 0])
    if profiles.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        profiles_2d = pca.fit_transform(profiles)
    else:
        profiles_2d = profiles

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            ax1.scatter(
                profiles_2d[mask, 0],
                profiles_2d[mask, 1],
                c=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                label=f"Cluster {cluster_id}",
                alpha=0.6,
                s=30,
            )
    ax1.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)"
        if profiles.shape[1] > 2
        else "Feature 1"
    )
    ax1.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)"
        if profiles.shape[1] > 2
        else "Feature 2"
    )
    ax1.set_title("Clusters (PCA projection)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. PCA projection colored by date type
    ax2 = fig.add_subplot(gs[0, 1])
    for date_type in date_types.unique():
        mask = date_types == date_type
        if mask.sum() > 0:
            ax2.scatter(
                profiles_2d[mask, 0],
                profiles_2d[mask, 1],
                c=DATE_TYPE_COLORS.get(date_type, "gray"),
                label=DATE_TYPE_LABELS.get(date_type, date_type),
                alpha=0.6,
                s=30,
            )
    ax2.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)"
        if profiles.shape[1] > 2
        else "Feature 1"
    )
    ax2.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)"
        if profiles.shape[1] > 2
        else "Feature 2"
    )
    ax2.set_title("Date Types (PCA projection)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    cm = metrics["confusion_matrix"]
    date_type_order = ["WD", "SA", "SU", "HO"]

    # Create labeled confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=[DATE_TYPE_LABELS.get(dt, dt) for dt in date_type_order[: cm.shape[0]]],
        columns=[f"Cluster {i}" for i in range(cm.shape[1])],
    )
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax3,
        cbar_kws={"label": "Count"},
    )
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Date Type")
    ax3.set_title("Cluster vs Date Type")

    # 4. Cluster purities
    ax4 = fig.add_subplot(gs[1, 0])
    purities = metrics["cluster_purities"]
    cluster_ids = sorted(purities.keys())
    purity_values = [purities[cid]["purity"] for cid in cluster_ids]
    most_common_types = [purities[cid]["most_common_type"] for cid in cluster_ids]

    bars = ax4.bar(
        [f"Cluster {cid}" for cid in cluster_ids],
        purity_values,
        color=[CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] for cid in cluster_ids],
    )
    ax4.set_ylabel("Purity")
    ax4.set_title("Cluster Purity\n(proportion of most common date type)")
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis="y")

    # Add labels on bars
    for i, (bar, dt) in enumerate(zip(bars, most_common_types)):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            DATE_TYPE_LABELS.get(dt, dt),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 5. Date type heterogeneity (entropy)
    ax5 = fig.add_subplot(gs[1, 1])
    heterogeneity = metrics["date_type_heterogeneity"]
    date_types_ordered = ["WD", "SA", "SU", "HO"]
    entropies = [
        heterogeneity.get(dt, {}).get("entropy", 0) for dt in date_types_ordered
    ]
    colors = [DATE_TYPE_COLORS.get(dt, "gray") for dt in date_types_ordered]

    bars = ax5.bar(
        [DATE_TYPE_LABELS.get(dt, dt) for dt in date_types_ordered],
        entropies,
        color=colors,
    )
    ax5.set_ylabel("Entropy (higher = more spread)")
    ax5.set_title("Date Type Heterogeneity\n(how spread across clusters)")
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. Cluster size distribution
    ax6 = fig.add_subplot(gs[1, 2])
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    ax6.bar(
        [f"Cluster {i}" for i in cluster_counts.index],
        cluster_counts.values,
        color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in cluster_counts.index],
    )
    ax6.set_ylabel("Number of Days")
    ax6.set_title("Cluster Sizes")
    ax6.grid(True, alpha=0.3, axis="y")

    # 7. Mean profiles by cluster
    ax7 = fig.add_subplot(gs[2, :])
    time_cols = [f"t_{400 + i * 15}" for i in range((2300 - 400) // 15 + 1)]
    time_cols = sort_time_columns(time_cols)
    hours = [
        int(col.replace("t_", "")) / 100.0 for col in time_cols[: profiles.shape[1]]
    ]

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            cluster_profiles = profiles[mask]
            mean_profile = cluster_profiles.mean(axis=0)
            std_profile = cluster_profiles.std(axis=0)

            ax7.plot(
                hours,
                mean_profile,
                color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                label=f"Cluster {cluster_id} (n={mask.sum()})",
                linewidth=2,
            )
            ax7.fill_between(
                hours,
                mean_profile - std_profile,
                mean_profile + std_profile,
                color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                alpha=0.2,
            )

    ax7.set_xlabel("Hour of Day")
    ax7.set_ylabel("Mean Count")
    ax7.set_title("Mean Daily Profiles by Cluster (±1 std)")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Overall title
    title = f"Clustering Analysis: k={n_clusters}"
    if station_code:
        title += f" - Station {station_code}"
        if station_name:
            title += f" ({station_name})"
    title += f"\nARI = {metrics['adjusted_rand_index']:.3f}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Save figure
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"clustering_{n_clusters}clusters"
        if station_code:
            filename += f"_{station_code}"
        filename += f"_{count_type}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"💾 Saved plot to {filepath}")

    plt.close()


def run_clustering_analysis(
    count_type: Literal["in", "out"] = "in",
    n_clusters: int = 3,
    station_codes: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    normalize: bool = True,
    random_state: int = 42,
):
    """
    Run clustering analysis on daily profiles.

    Args:
        count_type: Type of counts to analyze ("in" or "out")
        n_clusters: Number of clusters (default: 3)
        station_codes: Optional list of station codes. If None, analyzes all stations.
        output_dir: Output directory for plots and results
        normalize: Whether to normalize profiles before clustering
        random_state: Random seed for reproducibility
    """
    # Load parameters
    params_file = Path("src/workflow/params.json")
    with open(params_file) as f:
        params = json.load(f)

    time_min = params.get("step4", {}).get("time_min", 400)
    time_max = params.get("step4", {}).get("time_max", 2300)
    time_step = params.get("step4", {}).get("time_step", 15)

    # Load data
    print(f"📊 Loading {count_type} data...")
    data = load_data(
        include_checkins=(count_type == "in"),
        include_checkouts=(count_type == "out"),
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
    )

    # Map count_type to the correct key from load_data
    key_map = {"in": "checkins", "out": "checkouts"}
    df = data[key_map[count_type]]

    if df.empty:
        raise ValueError(f"Empty DataFrame for {count_type}")

    # Get unique stations
    if station_codes is None:
        station_codes = df["station_code"].unique().tolist()

    print(f"🔍 Analyzing {len(station_codes)} stations...")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/clustering_results")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Analyze each station separately
    for station_code in station_codes:
        station_data = df[df["station_code"] == station_code]
        if station_data.empty:
            continue

        station_name = station_data["station_name"].iloc[0]
        print(f"\n📈 Processing station: {station_code} - {station_name}")

        # Extract daily profiles
        profiles, date_types, metadata = extract_daily_profiles(
            station_data, station_code=station_code
        )

        if len(profiles) == 0:
            print(f"⚠️  No profiles for station {station_code}, skipping...")
            continue

        print(f"   Found {len(profiles)} daily profiles")

        # Cluster profiles
        cluster_labels, kmeans = cluster_profiles(
            profiles,
            n_clusters=n_clusters,
            random_state=random_state,
            normalize=normalize,
        )

        # Compute alignment metrics
        metrics = compute_alignment_metrics(cluster_labels, date_types)

        # Print summary
        print(f"   Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
        print("   Cluster purities:")
        for cid, info in metrics["cluster_purities"].items():
            print(
                f"      Cluster {cid}: {info['purity']:.2%} {DATE_TYPE_LABELS.get(info['most_common_type'], info['most_common_type'])} (n={info['size']})"
            )
        print("   Date type heterogeneity:")
        for dt, info in metrics["date_type_heterogeneity"].items():
            print(
                f"      {DATE_TYPE_LABELS.get(dt, dt)}: entropy={info['entropy']:.3f}, spread across {info['n_clusters']} clusters"
            )

        # Plot results
        plot_clustering_results(
            profiles,
            cluster_labels,
            date_types,
            metrics,
            station_code=station_code,
            station_name=station_name,
            output_dir=output_dir,
            count_type=count_type,
            n_clusters=n_clusters,
        )

        # Store results
        all_results.append(
            {
                "station_code": station_code,
                "station_name": station_name,
                "n_profiles": len(profiles),
                "ari": metrics["adjusted_rand_index"],
                "cluster_purities": metrics["cluster_purities"],
                "date_type_heterogeneity": metrics["date_type_heterogeneity"],
            }
        )

    # Save summary results
    summary_df = pd.DataFrame(
        [
            {
                "station_code": r["station_code"],
                "station_name": r["station_name"],
                "n_profiles": r["n_profiles"],
                "ari": r["ari"],
                **{
                    f"cluster_{cid}_purity": info["purity"]
                    for cid, info in r["cluster_purities"].items()
                },
                **{
                    f"{dt}_entropy": info["entropy"]
                    for dt, info in r["date_type_heterogeneity"].items()
                },
            }
            for r in all_results
        ]
    )

    summary_file = (
        output_dir / f"clustering_summary_{n_clusters}clusters_{count_type}.csv"
    )
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Saved summary to {summary_file}")

    # Print overall statistics
    if len(all_results) > 0:
        print(f"\n📊 Overall Statistics (k={n_clusters}):")
        print(f"   Mean ARI: {summary_df['ari'].mean():.3f}")
        print(f"   Median ARI: {summary_df['ari'].median():.3f}")

        # Mean HO entropy
        ho_entropy_mean = (
            f"{summary_df['HO_entropy'].mean():.3f}"
            if "HO_entropy" in summary_df.columns
            else "N/A"
        )
        print(f"   Mean HO entropy: {ho_entropy_mean}")

        # Mean WD entropy
        wd_entropy_mean = (
            f"{summary_df['WD_entropy'].mean():.3f}"
            if "WD_entropy" in summary_df.columns
            else "N/A"
        )
        print(f"   Mean WD entropy: {wd_entropy_mean}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster daily profiles and compare to day-type labels"
    )
    parser.add_argument(
        "--count-type",
        type=str,
        choices=["in", "out"],
        default="in",
        help="Type of counts to analyze (default: in)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters (default: 3)",
    )
    parser.add_argument(
        "--stations",
        type=str,
        nargs="+",
        help="Specific station codes to analyze (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: src/workflow/clustering_results)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize profiles before clustering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    run_clustering_analysis(
        count_type=args.count_type,
        n_clusters=args.n_clusters,
        station_codes=args.stations,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        normalize=not args.no_normalize,
        random_state=args.seed,
    )
