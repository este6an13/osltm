"""
Shape-Based Clustering of Station Profiles

This script performs shape-based clustering on station mean arrival profiles.
- Computes mean profiles for each station (fixed day type)
- Normalizes by total (shape only, removes scale)
- Applies Functional PCA (FPCA)
- Performs hierarchical clustering (Ward linkage)
- Visualizes clusters by plotting profiles grouped by cluster
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

from src.workflow.data_loader import load_data

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}


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


def time_columns_to_hours(time_cols: list[str]) -> np.ndarray:
    """
    Convert time column names to hours of day.

    Args:
        time_cols: List of time column names (should be sorted by numeric time value)

    Returns:
        Array of hours (e.g., 4.0, 4.25, 4.5, ...)
    """
    hours = []
    for col in time_cols:
        time_str = col.replace("t_", "")
        time_int = int(time_str)
        hour = time_int // 100
        minute = time_int % 100
        hours.append(hour + minute / 60.0)
    return np.array(hours)


def compute_mean_profiles_by_station(
    df: pd.DataFrame, day_type: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute mean arrival profiles for each station for a specific day type.

    Args:
        df: DataFrame from data_loader with columns: year, month, day, date_type,
            station_code, station_name, and time window columns (t_400, t_415, ..., t_2300)
        day_type: Day type to filter by (WD, SA, SU, HO)

    Returns:
        Tuple of (mean_profiles_df, station_names_series) where:
            - mean_profiles_df: DataFrame with rows as stations and columns as time windows
            - station_names_series: Series mapping station_code to station_name
    """
    # Filter by day type
    daytype_df = df[df["date_type"] == day_type].copy()

    if daytype_df.empty:
        return pd.DataFrame(), pd.Series(dtype=str)

    # Get time window columns (all columns starting with 't_')
    time_cols = [col for col in daytype_df.columns if col.startswith("t_")]
    time_cols = sort_time_columns(time_cols)

    # Group by station and compute mean
    mean_profiles = []
    station_info = {}

    for station_code in daytype_df["station_code"].unique():
        station_data = daytype_df[daytype_df["station_code"] == station_code]

        # Get station name (should be the same for all rows)
        station_name = station_data["station_name"].iloc[0]
        station_info[station_code] = station_name

        # Extract time series for this station
        time_series = station_data[time_cols].copy()

        # Fill NaN with 0
        time_series = time_series.fillna(0)

        # Compute mean profile
        mean_profile = time_series.mean(axis=0).values

        # Store as dictionary for DataFrame creation
        profile_dict = {time_cols[i]: mean_profile[i] for i in range(len(time_cols))}
        profile_dict["station_code"] = station_code
        mean_profiles.append(profile_dict)

    if not mean_profiles:
        return pd.DataFrame(), pd.Series(dtype=str)

    # Create DataFrame with stations as rows and time columns as columns
    mean_profiles_df = pd.DataFrame(mean_profiles)
    mean_profiles_df = mean_profiles_df.set_index("station_code")

    # Ensure columns are in correct order
    mean_profiles_df = mean_profiles_df[time_cols]

    # Create station names mapping
    station_names_series = pd.Series(station_info)

    return mean_profiles_df, station_names_series


def normalize_by_total(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize profiles by their total (sum to 1).

    Args:
        profiles_df: DataFrame with rows as stations and columns as time windows

    Returns:
        Normalized DataFrame (each row sums to 1)
    """
    normalized = profiles_df.copy()
    row_sums = normalized.sum(axis=1)
    # Avoid division by zero
    row_sums = row_sums.replace(0, 1)
    normalized = normalized.div(row_sums, axis=0)
    return normalized


def perform_fpca(
    profiles_df: pd.DataFrame, n_components: Optional[int] = None
) -> tuple[np.ndarray, PCA]:
    """
    Perform Functional PCA on normalized profiles.

    Args:
        profiles_df: DataFrame with rows as stations and columns as time windows
        n_components: Number of components (if None, uses all)

    Returns:
        Tuple of (projected_data, pca_model)
    """
    X = profiles_df.values

    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(X)

    return projected, pca


def perform_hierarchical_clustering(
    data: np.ndarray, n_clusters: int, method: str = "ward"
) -> np.ndarray:
    """
    Perform hierarchical clustering using Ward linkage.

    Args:
        data: Array of shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        method: Linkage method (default: "ward")

    Returns:
        Array of cluster labels
    """
    Z = linkage(data, method=method)
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    return labels


def plot_cluster_profiles(
    profiles_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    station_names: pd.Series,
    day_type: str,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
    n_clusters: Optional[int] = None,
) -> None:
    """
    Plot station profiles grouped by cluster.

    Args:
        profiles_df: DataFrame with rows as stations and columns as time windows
        cluster_labels: Array of cluster labels for each station
        station_names: Series mapping station_code to station_name
        day_type: Day type being visualized
        output_dir: Optional directory to save the plot
        count_type: Type of counts ("checkins" or "checkouts")
        n_clusters: Number of clusters (for title)
    """
    if profiles_df.empty:
        print("⚠️  No data to plot")
        return

    time_cols = profiles_df.columns.tolist()
    hours = time_columns_to_hours(time_cols)

    if n_clusters is None:
        n_clusters = len(np.unique(cluster_labels))

    # Create subplots for each cluster
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Plot each cluster
    for cluster_id in range(1, n_clusters + 1):
        ax = axes[cluster_id - 1]
        cluster_mask = cluster_labels == cluster_id
        cluster_stations = profiles_df.index[cluster_mask]

        if len(cluster_stations) == 0:
            ax.set_visible(False)
            continue

        # Plot each station in the cluster
        for station_code in cluster_stations:
            profile = profiles_df.loc[station_code].values
            ax.plot(hours, profile, alpha=0.5, linewidth=1, label=station_code)

        # Plot cluster mean
        cluster_profiles = profiles_df.loc[cluster_stations].values
        cluster_mean = cluster_profiles.mean(axis=0)
        ax.plot(
            hours,
            cluster_mean,
            color="black",
            linewidth=3,
            label=f"Cluster {cluster_id} Mean",
        )

        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_ylabel("Normalized Count", fontsize=11)
        ax.set_title(
            f"Cluster {cluster_id} (n={len(cluster_stations)})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)

        # Format x-axis
        ax.set_xticks(range(int(hours[0]), int(hours[-1]) + 1, 2))
        ax.set_xticklabels(
            [f"{h:02d}:00" for h in range(int(hours[0]), int(hours[-1]) + 1, 2)]
        )

    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    # Add main title
    day_type_label = DATE_TYPE_LABELS.get(day_type, day_type)
    fig.suptitle(
        f"Shape-Based Clustering: Station Profiles ({day_type_label})\n"
        f"{count_type.capitalize()} - Normalized by Total, FPCA + Ward Clustering",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"cluster_shape_{day_type}_{count_type}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved plot to {filename}")

    plt.show()


def run_shape_clustering(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    day_type: Literal["WD", "SA", "SU", "HO"] = "WD",
    n_clusters: int = 4,
    n_components: Optional[int] = None,
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Run shape-based clustering on station profiles.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        day_type: Day type to analyze (WD, SA, SU, HO)
        n_clusters: Number of clusters to form
        n_components: Number of FPCA components (if None, uses all)
        output_dir: Directory to save plots (default: src/workflow/clustering_results)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all stations.

    Returns:
        Tuple of (profiles_df, cluster_labels)
    """
    # Load parameters
    if params_path is None:
        params_path = Path("src/workflow/params.json")

    with open(params_path) as f:
        params = json.load(f)

    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)

    # Load data
    print(f"📊 Loading {count_type} data...")
    if station_codes:
        print(f"   Filtering to {len(station_codes)} specified stations")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
    )

    if count_type not in data:
        raise ValueError(f"No {count_type} data available")

    df = data[count_type]

    if df.empty:
        raise ValueError(f"Empty DataFrame for {count_type}")

    # Filter by station codes if provided (validate which ones have data)
    if station_codes is not None:
        available_stations = set(df["station_code"].unique())
        requested_stations = set(station_codes)
        missing = requested_stations - available_stations
        if missing:
            print(
                f"⚠️  Warning: Some specified stations have no data: {sorted(missing)}"
            )
        station_codes = [sc for sc in station_codes if sc in available_stations]
        if not station_codes:
            raise ValueError("No data found for any of the specified stations")
        print(f"🔍 Filtered to {len(station_codes)} stations with data")
    else:
        print("🔍 Using all stations from data")

    # Filter by day type
    df = df[df["date_type"] == day_type].copy()
    if df.empty:
        raise ValueError(f"No data found for day type: {day_type}")

    print(f"🔍 Filtered to day type: {DATE_TYPE_LABELS.get(day_type, day_type)}")
    print(f"   Found {len(df)} (station, day) profiles")

    # Compute mean profiles
    print("\n📈 Computing mean profiles for each station...")
    mean_profiles_df, station_names = compute_mean_profiles_by_station(df, day_type)

    if mean_profiles_df.empty:
        raise ValueError("No mean profiles computed")

    print(f"   Computed profiles for {len(mean_profiles_df)} stations")

    # Normalize by total (shape only)
    print("\n🔄 Normalizing profiles by total (shape-based)...")
    normalized_profiles = normalize_by_total(mean_profiles_df)
    print("   Profiles normalized (each sums to 1)")

    # Perform FPCA
    print("\n📊 Performing Functional PCA...")
    projected, pca = perform_fpca(normalized_profiles, n_components=n_components)
    explained_variance = pca.explained_variance_ratio_
    print(
        f"   Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}"
    )
    if len(explained_variance) > 2:
        total_explained = sum(explained_variance)
        print(f"   Total explained variance: {total_explained:.2%}")

    # Perform hierarchical clustering
    print(f"\n🔗 Performing hierarchical clustering (Ward, {n_clusters} clusters)...")
    cluster_labels = perform_hierarchical_clustering(
        projected, n_clusters, method="ward"
    )
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   Cluster sizes: {dict(zip(unique_clusters, counts))}")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/clustering_results")
    output_dir = Path(output_dir)

    # Plot cluster profiles
    print("\n🎨 Plotting cluster profiles...")
    plot_cluster_profiles(
        normalized_profiles,
        cluster_labels,
        station_names,
        day_type,
        output_dir=output_dir,
        count_type=count_type,
        n_clusters=n_clusters,
    )

    print("\n✅ Completed shape-based clustering")
    return normalized_profiles, cluster_labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Shape-based clustering of station arrival profiles"
    )
    parser.add_argument(
        "--count_type",
        choices=["checkins", "checkouts"],
        default="checkins",
        help="Type of counts to analyze (default: checkins)",
    )
    parser.add_argument(
        "--day_type",
        choices=["WD", "SA", "SU", "HO"],
        default="WD",
        help="Day type to analyze (default: WD)",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=4,
        help="Number of clusters (default: 4)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Number of FPCA components (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/clustering_results",
        help="Directory to save plots (default: src/workflow/clustering_results)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="src/workflow/params.json",
        help="Path to params.json (default: src/workflow/params.json)",
    )
    parser.add_argument(
        "--stations",
        type=str,
        nargs="+",
        help="Optional list of station codes to analyze (default: all stations)",
    )

    args = parser.parse_args()

    results = run_shape_clustering(
        count_type=args.count_type,
        day_type=args.day_type,
        n_clusters=args.n_clusters,
        n_components=args.n_components,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
    )

    print(
        f"\n📊 Summary: Clustered {len(results[0])} stations into {args.n_clusters} clusters"
    )
