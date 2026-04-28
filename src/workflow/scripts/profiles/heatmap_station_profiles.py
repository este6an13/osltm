"""
Heatmap of Average Station Profiles

This script creates a heatmap showing mean arrival profiles for each station.
- Rows = stations
- Columns = time bins
- Values = mean arrivals (for a fixed day type)
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def time_columns_to_hours(time_cols: list[str]) -> list[str]:
    """
    Convert time column names to hour labels for display.

    Args:
        time_cols: List of time column names (e.g., ["t_400", "t_415"])

    Returns:
        List of hour labels (e.g., ["04:00", "04:15"])
    """
    labels = []
    for col in time_cols:
        time_str = col.replace("t_", "")
        time_int = int(time_str)
        hour = time_int // 100
        minute = time_int % 100
        labels.append(f"{hour:02d}:{minute:02d}")
    return labels


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


def plot_heatmap(
    mean_profiles_df: pd.DataFrame,
    station_names: pd.Series,
    day_type: str,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
    cmap: str = "YlOrRd",
    figsize: Optional[tuple[float, float]] = None,
    show_plots: bool = False,
) -> None:
    """
    Create a heatmap of station arrival profiles.

    Args:
        mean_profiles_df: DataFrame with rows as stations and columns as time windows
        station_names: Series mapping station_code to station_name
        day_type: Day type being visualized
        output_dir: Optional directory to save the plot
        count_type: Type of counts ("checkins" or "checkouts")
        cmap: Colormap for heatmap (default: "YlOrRd")
        figsize: Optional figure size tuple (width, height)
    """
    if mean_profiles_df.empty:
        print("⚠️  No data to plot")
        return

    # Prepare labels
    time_cols = mean_profiles_df.columns.tolist()
    time_labels = time_columns_to_hours(time_cols)

    # Create row labels with station code and name
    row_labels = [
        f"{idx}\n{station_names.get(idx, idx)}" for idx in mean_profiles_df.index
    ]

    # Determine figure size if not provided
    if figsize is None:
        n_stations = len(mean_profiles_df)
        n_time_bins = len(time_cols)
        # Adjust size based on data dimensions
        height = max(8, n_stations * 0.3)
        width = max(12, n_time_bins * 0.15)
        figsize = (width, height)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    mean_label = f"Mean {count_type.capitalize()}"
    sns.heatmap(
        mean_profiles_df,
        cmap=cmap,
        xticklabels=False,  # Don't show labels initially
        yticklabels=row_labels,
        cbar_kws={"label": mean_label},
        linewidths=0.5,
        linecolor="gray",
        rasterized=False,  # Set to True for very large datasets
        ax=ax,
    )

    # Set x-axis labels manually to ensure proper alignment
    # Show every 4th label if there are many time bins, otherwise show all
    if len(time_labels) > 20:
        step = 4
        tick_positions = list(range(0, len(time_labels), step))
        tick_labels = [time_labels[i] for i in tick_positions]
    else:
        tick_positions = list(range(len(time_labels)))
        tick_labels = time_labels

    # Set the tick positions and labels
    ax.set_xticks([i + 0.5 for i in tick_positions])  # +0.5 to center on bins
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Set title
    day_type_label = DATE_TYPE_LABELS.get(day_type, day_type)
    title_text = (
        f"Station Arrival Profiles Heatmap ({day_type_label})\n"
        "Rows = Stations, Columns = Time Bins"
    )
    fig.suptitle(
        title_text,
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Adjust layout
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"heatmap_stations_{day_type}_{count_type}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved plot to {filename}")

    if show_plots:
        plt.show()
    plt.close(fig)


def run_heatmap_analysis(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    day_type: Literal["WD", "SA", "SU", "HO"] = "WD",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    cmap: str = "YlOrRd",
    figsize: Optional[tuple[float, float]] = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Run heatmap analysis for station profiles.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        day_type: Day type to analyze (WD, SA, SU, HO)
        output_dir: Directory to save plots (default: src/workflow/results/heatmap_results)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all stations.
        cmap: Colormap for heatmap (default: "YlOrRd")
        figsize: Optional figure size tuple (width, height)
        show_plots: Whether to show the plot interactively

    Returns:
        DataFrame with mean profiles (stations x time bins)
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
    print(f"   Time bins: {len(mean_profiles_df.columns)}")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/results/heatmap_results")
    output_dir = Path(output_dir)

    # Plot heatmap
    print("\n🎨 Creating heatmap...")
    plot_heatmap(
        mean_profiles_df,
        station_names,
        day_type,
        output_dir=output_dir,
        count_type=count_type,
        cmap=cmap,
        figsize=figsize,
        show_plots=show_plots,
    )

    # Save mean profiles to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_df = mean_profiles_df.copy()
    csv_df.insert(0, "station_name", csv_df.index.map(station_names))
    csv_path = output_dir / f"heatmap_profiles_{day_type}_{count_type}.csv"
    csv_df.to_csv(csv_path, index=True)
    print(f"💾 Saved mean profiles to {csv_path}")

    print("\n✅ Completed heatmap analysis")
    return mean_profiles_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create heatmap of station arrival profiles"
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
        "--output_dir",
        type=str,
        default="src/workflow/results/heatmap_results",
        help="Directory to save plots (default: src/workflow/results/heatmap_results)",
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
        "--cmap",
        type=str,
        default="YlOrRd",
        help="Colormap for heatmap (default: YlOrRd). Examples: viridis, plasma, YlOrRd, RdBu_r",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size as width height (default: auto)",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display the plots interactively in a window (blocks execution)",
    )

    args = parser.parse_args()

    figsize = None
    if args.figsize:
        figsize = tuple(args.figsize)

    results = run_heatmap_analysis(
        count_type=args.count_type,
        day_type=args.day_type,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
        cmap=args.cmap,
        figsize=figsize,
        show_plots=args.show_plots,
    )

    print(f"\n📊 Summary: Created heatmap for {len(results)} stations")
