"""
Functional PCA Across Stations

This script performs Functional Principal Component Analysis (FPCA) on all (station, day) profiles
together. Each observation is a (station, day) profile (a vector of counts across time windows).

The results are projected onto 2 principal components and visualized with:
- Scatter plot of PC1 vs PC2
- Color = station
- Facet by day type
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.workflow.data_loader import load_data

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

# Order for plotting (to ensure consistent facet order)
DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]


def extract_all_profiles(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Extract all (station, day) profiles from the data.

    Args:
        df: DataFrame from data_loader with columns: year, month, day, date_type,
            station_code, station_name, and time window columns (t_400, t_415, ..., t_2300)

    Returns:
        Tuple of (time_series_matrix, station_codes, station_names, date_types, date_strs) where:
            - time_series_matrix: DataFrame with rows as (station, day) and columns as time windows
            - station_codes: Series with station_code for each row
            - station_names: Series with station_name for each row
            - date_types: Series with date_type for each row
            - date_strs: Series with date strings (YYYY-MM-DD) for each row
    """
    if df.empty:
        return (
            pd.DataFrame(),
            pd.Series(dtype=str),
            pd.Series(dtype=str),
            pd.Series(dtype=str),
            pd.Series(dtype=str),
        )

    # Get time window columns (all columns starting with 't_')
    time_cols = [col for col in df.columns if col.startswith("t_")]
    time_cols = sorted(time_cols)  # Ensure proper ordering

    # Extract time series matrix (each row is a (station, day), columns are time windows)
    time_series = df[time_cols].copy()

    # Extract metadata
    station_codes = df["station_code"].copy()
    station_names = df["station_name"].copy()
    date_types = df["date_type"].copy()

    # Create date identifier for reference
    date_strs = (
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(str).str.zfill(2)
    )

    return time_series, station_codes, station_names, date_types, date_strs


def perform_fpca(
    time_series: pd.DataFrame,
    n_components: int = 2,
    standardize: bool = True,
) -> tuple[np.ndarray, PCA, Optional[StandardScaler], pd.Index]:
    """
    Perform Functional PCA on time series data.

    Args:
        time_series: DataFrame with rows as (station, day) and columns as time windows
        n_components: Number of principal components to extract
        standardize: Whether to standardize the data before PCA

    Returns:
        Tuple of (projected_data, pca_model, scaler, kept_indices) where kept_indices
        are the indices of rows that were kept after filtering
    """
    if time_series.empty:
        return np.array([]), None, None, pd.Index([])

    # Separate time window columns from metadata
    time_cols = [col for col in time_series.columns if col.startswith("t_")]
    time_data = time_series[time_cols].copy()

    # Remove rows with all NaN
    time_data = time_data.dropna(how="all")
    kept_indices = time_data.index

    if len(time_data) < n_components:
        print(f"⚠️  Not enough samples ({len(time_data)}) for {n_components} components")
        return np.array([]), None, None, pd.Index([])

    # Fill remaining NaN with 0 (or could use forward fill, mean, etc.)
    time_data = time_data.fillna(0)

    # Standardize if requested
    scaler = None
    if standardize:
        scaler = StandardScaler()
        time_data_scaled = scaler.fit_transform(time_data)
    else:
        time_data_scaled = time_data.values

    # Perform PCA
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(time_data_scaled)

    return projected, pca, scaler, kept_indices


def plot_fpca_results(
    projected: np.ndarray,
    station_codes: pd.Series,
    station_names: pd.Series,
    date_types: pd.Series,
    date_strs: pd.Series,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
    day_types: Optional[list[str]] = None,
) -> None:
    """
    Plot FPCA results with points colored by station and faceted by day type.

    Args:
        projected: 2D array of projected data (n_samples, 2)
        station_codes: Series with station_code for each sample
        station_names: Series with station_name for each sample
        date_types: Series with date_type for each sample
        date_strs: Series with date strings for annotation
        output_dir: Optional directory to save the plot
        count_type: Type of counts ("checkins" or "checkouts")
        day_types: Optional list of day types to include (WD, SA, SU, HO). If None, uses all.
    """
    if len(projected) == 0:
        print("⚠️  No data to plot")
        return

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(
        {
            "PC1": projected[:, 0],
            "PC2": projected[:, 1],
            "station_code": station_codes.values,
            "station_name": station_names.values,
            "date_type": date_types.values,
            "date_str": date_strs.values,
        }
    )

    # Filter by day types if provided
    if day_types is not None:
        plot_df = plot_df[plot_df["date_type"].isin(day_types)].copy()
        if plot_df.empty:
            print("⚠️  No data for specified day types")
            return
        day_type_order = [dt for dt in DATE_TYPE_ORDER if dt in day_types]
    else:
        day_type_order = DATE_TYPE_ORDER

    # Map date types to labels
    plot_df["day_type_label"] = plot_df["date_type"].map(DATE_TYPE_LABELS)

    # Get unique stations and assign colors
    unique_stations = sorted(plot_df["station_code"].unique())
    n_stations = len(unique_stations)

    # Use a colormap that works well for many stations
    if n_stations <= 10:
        # Use distinct colors for small number of stations
        colors = sns.color_palette("tab10", n_stations)
    else:
        # Use a continuous colormap for many stations
        colors = sns.color_palette("husl", n_stations)

    station_color_map = dict(zip(unique_stations, colors))

    # Create faceted plot
    col_order = [DATE_TYPE_LABELS.get(dt, dt) for dt in day_type_order]
    g = sns.FacetGrid(
        plot_df,
        col="day_type_label",
        col_order=col_order,
        col_wrap=2 if len(day_type_order) > 1 else 1,
        height=6,
        aspect=1.2,
        sharex=True,
        sharey=True,
    )

    def scatter_facet(data, **kwargs):
        ax = plt.gca()
        for station_code in data["station_code"].unique():
            station_data = data[data["station_code"] == station_code]
            color = station_color_map[station_code]
            ax.scatter(
                station_data["PC1"],
                station_data["PC2"],
                c=color,
                label=station_code,
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidths=0.3,
            )

    g.map_dataframe(scatter_facet)

    # Set labels and titles
    g.set_axis_labels(
        "First Principal Component (PC1)", "Second Principal Component (PC2)"
    )
    g.set_titles("{col_name}")

    # Add main title
    g.fig.suptitle(
        f"Functional PCA Across Stations ({count_type.capitalize()})\n"
        f"Color = Station, Facet = Day Type",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Add legend (only show first few stations to avoid clutter)
    # Or create a separate legend if needed
    if n_stations <= 15:
        # Add legend to the last subplot
        handles, labels = g.axes[-1].get_legend_handles_labels()
        if handles:
            g.axes[-1].legend(
                handles,
                labels,
                title="Station",
                loc="upper right",
                bbox_to_anchor=(1.15, 1),
                fontsize=8,
                ncol=1 if n_stations <= 10 else 2,
            )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"fpca_across_stations_{count_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved plot to {filename}")

    plt.show()


def run_fpca_across_stations(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    n_components: int = 2,
    standardize: bool = True,
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
) -> dict:
    """
    Run Functional PCA across all stations.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        n_components: Number of principal components (default: 2)
        standardize: Whether to standardize data before PCA (default: True)
        output_dir: Directory to save plots (default: src/workflow/results/fpca_results)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all stations.
        day_types: Optional list of day types to include (WD, SA, SU, HO). If None, uses all.

    Returns:
        Dictionary with FPCA results
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

    # Filter by day types if provided
    if day_types is not None:
        df = df[df["date_type"].isin(day_types)].copy()
        if df.empty:
            raise ValueError(f"No data found for specified day types: {day_types}")
        print(f"🔍 Filtered to day types: {day_types}")
    else:
        print("🔍 Using all day types from data")

    print(f"   Found {len(df)} (station, day) profiles")

    # Extract all profiles
    time_series, station_codes_extracted, station_names, date_types, date_strs = (
        extract_all_profiles(df)
    )

    if time_series.empty:
        raise ValueError("No time series data extracted")

    # time_series already contains only time window columns
    time_data = time_series.copy()

    print(
        f"   Processing {len(time_data)} profiles across {len(station_codes_extracted.unique())} stations"
    )

    # Perform FPCA (will filter all-NaN rows internally)
    projected, pca, scaler, kept_indices = perform_fpca(
        time_data, n_components=n_components, standardize=standardize
    )

    if len(projected) == 0:
        raise ValueError("FPCA failed - no projected data")

    # Filter metadata to match kept indices
    station_codes_filtered = station_codes_extracted.loc[kept_indices].copy()
    station_names = station_names.loc[kept_indices].copy()
    date_types = date_types.loc[kept_indices].copy()
    date_strs = date_strs.loc[kept_indices].copy()

    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print("\n📈 FPCA Results:")
    print(
        f"   Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}"
    )
    if n_components > 2:
        total_explained = sum(explained_variance)
        print(f"   Total explained variance: {total_explained:.2%}")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/results/fpca_results")
    output_dir = Path(output_dir)

    # Plot results
    plot_fpca_results(
        projected,
        station_codes_filtered,
        station_names,
        date_types,
        date_strs,
        output_dir=output_dir,
        count_type=count_type,
        day_types=day_types,
    )

    # Save scores CSV
    scores_df = pd.DataFrame(
        {
            "station_code": station_codes_filtered.values,
            "station_name": station_names.values,
            "date_type": date_types.values,
            "date_str": date_strs.values,
        }
    )
    for i in range(projected.shape[1]):
        scores_df[f"PC{i + 1}"] = projected[:, i]

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / f"fpca_across_stations_{count_type}.csv"
    scores_df.to_csv(scores_csv, index=False)
    print(f"💾 Saved scores to {scores_csv}")

    # Save explained variance CSV
    variance_rows = []
    cumulative = 0.0
    for i, ev in enumerate(explained_variance):
        cumulative += ev
        variance_rows.append(
            {
                "component": f"PC{i + 1}",
                "explained_variance_ratio": ev,
                "cumulative_explained_variance": cumulative,
            }
        )
    variance_df = pd.DataFrame(variance_rows)
    variance_csv = output_dir / f"fpca_across_stations_variance_{count_type}.csv"
    variance_df.to_csv(variance_csv, index=False)
    print(f"💾 Saved explained variance to {variance_csv}")

    # Prepare results
    results = {
        "projected": projected,
        "station_codes": station_codes_filtered.values,
        "station_names": station_names.values,
        "date_types": date_types.values,
        "date_strs": date_strs.values,
        "explained_variance": explained_variance,
        "pca": pca,
        "scaler": scaler,
    }

    print("\n✅ Completed FPCA across stations")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform Functional PCA across all stations on (station, day) profiles"
    )
    parser.add_argument(
        "--count_type",
        choices=["checkins", "checkouts"],
        default="checkins",
        help="Type of counts to analyze (default: checkins)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of principal components (default: 2)",
    )
    parser.add_argument(
        "--no_standardize",
        action="store_true",
        help="Don't standardize data before PCA",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/fpca_results",
        help="Directory to save plots (default: src/workflow/results/fpca_results)",
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
    parser.add_argument(
        "--day_types",
        type=str,
        nargs="+",
        choices=["WD", "SA", "SU", "HO"],
        help="Optional list of day types to include (WD, SA, SU, HO). Default: all.",
    )

    args = parser.parse_args()

    results = run_fpca_across_stations(
        count_type=args.count_type,
        n_components=args.n_components,
        standardize=not args.no_standardize,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
        day_types=args.day_types,
    )

    print(f"\n📊 Summary: Analyzed {len(results['projected'])} (station, day) profiles")
