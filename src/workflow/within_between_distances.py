"""
Within- vs Between-Group Distance Ratios

This script computes the ratio of within-group to between-group distances for daily count profiles,
grouped by day type (WD, SA, SU, HO). This metric helps assess how well-separated different day types
are in the functional space.

For each station:
- Computes average L2/Euclidean distance within same day-type groups
- Computes average L2/Euclidean distance between different day-type groups
- Calculates ratio R = mean(within-group distance) / mean(between-group distance)

Lower ratios indicate better separation between day types.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

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


def extract_time_series_data(
    df: pd.DataFrame, station_code: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract time series data for a specific station.

    Args:
        df: DataFrame from data_loader with columns: year, month, day, date_type,
            station_code, and time window columns (t_400, t_415, ..., t_2300)
        station_code: Station code to filter by

    Returns:
        Tuple of (time_series_matrix, date_types) where:
            - time_series_matrix: DataFrame with rows as days and columns as time windows
            - date_types: Series with date_type for each day
    """
    # Filter by station
    station_df = df[df["station_code"] == station_code].copy()

    if station_df.empty:
        return pd.DataFrame(), pd.Series(dtype=str)

    # Get time window columns (all columns starting with 't_')
    time_cols = [col for col in station_df.columns if col.startswith("t_")]
    time_cols = sorted(time_cols)  # Ensure proper ordering

    # Extract time series matrix (each row is a day, columns are time windows)
    time_series = station_df[time_cols].copy()

    # Extract date types
    date_types = station_df["date_type"].copy()

    return time_series, date_types


def compute_distance_ratios(time_series: pd.DataFrame, date_types: pd.Series) -> dict:
    """
    Compute within-group and between-group distance ratios.

    Args:
        time_series: DataFrame with rows as days and columns as time windows
        date_types: Series with date_type for each day

    Returns:
        Dictionary with:
            - within_distances: List of within-group distances
            - between_distances: List of between-group distances
            - ratio: R = mean(within) / mean(between)
            - within_mean: Mean within-group distance
            - between_mean: Mean between-group distance
            - n_within: Number of within-group pairs
            - n_between: Number of between-group pairs
            - by_group: Dict with ratios for each day type pair
    """
    if time_series.empty or len(time_series) < 2:
        return {
            "ratio": np.nan,
            "within_mean": np.nan,
            "between_mean": np.nan,
            "n_within": 0,
            "n_between": 0,
            "within_distances": [],
            "between_distances": [],
            "by_group": {},
        }

    # Fill NaN with 0
    time_series = time_series.fillna(0)

    # Convert to numpy array
    X = time_series.values

    # Compute pairwise Euclidean distances
    distances = squareform(pdist(X, metric="euclidean"))

    # Get unique day types
    unique_types = date_types.unique()

    within_distances = []
    between_distances = []
    by_group = {}

    # Compute within-group distances
    for day_type in unique_types:
        mask = date_types == day_type
        indices = np.where(mask)[0]

        if len(indices) < 2:
            continue

        # Get pairwise distances within this group
        group_distances = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dist = distances[indices[i], indices[j]]
                group_distances.append(dist)
                within_distances.append(dist)

        if group_distances:
            by_group[f"{day_type}_within"] = {
                "mean": np.mean(group_distances),
                "std": np.std(group_distances),
                "count": len(group_distances),
            }

    # Compute between-group distances
    for i, type1 in enumerate(unique_types):
        for j, type2 in enumerate(unique_types):
            if i >= j:  # Avoid duplicates and same-type pairs
                continue

            mask1 = date_types == type1
            mask2 = date_types == type2
            indices1 = np.where(mask1)[0]
            indices2 = np.where(mask2)[0]

            if len(indices1) == 0 or len(indices2) == 0:
                continue

            # Get pairwise distances between these groups
            group_distances = []
            for idx1 in indices1:
                for idx2 in indices2:
                    dist = distances[idx1, idx2]
                    group_distances.append(dist)
                    between_distances.append(dist)

            if group_distances:
                by_group[f"{type1}_vs_{type2}"] = {
                    "mean": np.mean(group_distances),
                    "std": np.std(group_distances),
                    "count": len(group_distances),
                }

    # Compute overall ratio
    within_mean = np.mean(within_distances) if within_distances else np.nan
    between_mean = np.mean(between_distances) if between_distances else np.nan
    ratio = within_mean / between_mean if between_mean > 0 else np.nan

    return {
        "ratio": ratio,
        "within_mean": within_mean,
        "between_mean": between_mean,
        "n_within": len(within_distances),
        "n_between": len(between_distances),
        "within_distances": within_distances,
        "between_distances": between_distances,
        "by_group": by_group,
    }


def plot_distance_distributions(
    results: dict,
    station_code: str,
    station_name: str,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot distributions of within-group and between-group distances.

    Args:
        results: Results dictionary from compute_distance_ratios
        station_code: Station code
        station_name: Station name
        output_dir: Optional directory to save the plot
        count_type: Type of counts ("checkins" or "checkouts")
    """
    if not results["within_distances"] or not results["between_distances"]:
        print(f"⚠️  No data to plot for station {station_code}")
        return

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])  # Histogram
    ax2 = fig.add_subplot(gs[1])  # Box plot
    ax3 = fig.add_subplot(gs[2])  # Statistics text

    # Plot 1: Histogram of distances
    ax1.hist(
        results["within_distances"],
        bins=30,
        alpha=0.6,
        label="Within-group",
        color="#2E86AB",
        edgecolor="black",
    )
    ax1.hist(
        results["between_distances"],
        bins=30,
        alpha=0.6,
        label="Between-group",
        color="#F18F01",
        edgecolor="black",
    )
    ax1.axvline(
        results["within_mean"],
        color="#2E86AB",
        linestyle="--",
        linewidth=2,
        label=f"Within mean: {results['within_mean']:.2f}",
    )
    ax1.axvline(
        results["between_mean"],
        color="#F18F01",
        linestyle="--",
        linewidth=2,
        label=f"Between mean: {results['between_mean']:.2f}",
    )
    ax1.set_xlabel("Euclidean Distance", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distance Distributions", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Enhanced Box plot
    data_to_plot = [results["within_distances"], results["between_distances"]]
    bp = ax2.boxplot(
        data_to_plot,
        labels=["Within-group", "Between-group"],
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanline=True,
    )

    # Customize box colors
    bp["boxes"][0].set_facecolor("#2E86AB")
    bp["boxes"][1].set_facecolor("#F18F01")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_alpha(0.7)
    bp["boxes"][0].set_edgecolor("black")
    bp["boxes"][1].set_edgecolor("black")
    bp["boxes"][0].set_linewidth(1.5)
    bp["boxes"][1].set_linewidth(1.5)

    # Customize median lines
    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)

    # Customize mean lines
    for mean in bp["means"]:
        mean.set_color("green")
        mean.set_linewidth(2)
        mean.set_linestyle("--")

    # Customize whiskers and caps
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(1.5)
    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(1.5)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor("gray")
        flier.set_markeredgecolor("black")
        flier.set_alpha(0.5)

    ax2.set_ylabel("Euclidean Distance", fontsize=12)
    ax2.set_title("Distance Comparison (Boxplot)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add legend for boxplot elements
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="#2E86AB", lw=4, alpha=0.7, label="Within-group"),
        Line2D([0], [0], color="#F18F01", lw=4, alpha=0.7, label="Between-group"),
        Line2D([0], [0], color="red", lw=2, label="Median"),
        Line2D([0], [0], color="green", lw=2, linestyle="--", label="Mean"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Plot 3: Statistics text panel
    ax3.axis("off")
    ratio_text = "📊 Statistics\n\n"
    ratio_text += f"Ratio R = {results['ratio']:.3f}\n\n"
    ratio_text += "Within-group:\n"
    ratio_text += f"  Mean: {results['within_mean']:.2f}\n"
    ratio_text += f"  Std: {np.std(results['within_distances']):.2f}\n"
    ratio_text += f"  Median: {np.median(results['within_distances']):.2f}\n"
    ratio_text += f"  Count: {results['n_within']}\n\n"
    ratio_text += "Between-group:\n"
    ratio_text += f"  Mean: {results['between_mean']:.2f}\n"
    ratio_text += f"  Std: {np.std(results['between_distances']):.2f}\n"
    ratio_text += f"  Median: {np.median(results['between_distances']):.2f}\n"
    ratio_text += f"  Count: {results['n_between']}\n\n"

    # Interpretation
    if results["ratio"] < 0.5:
        interpretation = "✅ Strong separation"
    elif results["ratio"] < 0.8:
        interpretation = "✓ Moderate separation"
    elif results["ratio"] < 1.2:
        interpretation = "⚠ Weak separation"
    else:
        interpretation = "❌ Poor separation"

    ratio_text += f"Interpretation:\n{interpretation}"

    ax3.text(
        0.1,
        0.95,
        ratio_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7, pad=10),
    )

    fig.suptitle(
        f"Within vs Between-Group Distances: {station_code} - {station_name}\n({count_type.capitalize()})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"distances_{station_code}_{count_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved plot to {filename}")

    plt.show()


def run_distance_analysis(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Run within- vs between-group distance analysis for each station.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        output_dir: Directory to save plots (default: src/workflow/distance_results)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all from persistence.
        plot: Whether to generate plots (default: True)

    Returns:
        DataFrame with results for each station
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

    # Get unique stations (if not already specified)
    if station_codes is None:
        station_codes = df["station_code"].unique().tolist()
    else:
        # Filter to only stations that actually have data
        available_stations = set(df["station_code"].unique())
        requested_stations = set(station_codes)
        missing = requested_stations - available_stations
        if missing:
            print(
                f"⚠️  Warning: Some specified stations have no data: {sorted(missing)}"
            )
        station_codes = [sc for sc in station_codes if sc in available_stations]

    print(f"🔍 Analyzing {len(station_codes)} stations...")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/distance_results")
    output_dir = Path(output_dir)

    results_list = []

    # Process each station
    for station_code in station_codes:
        station_df = df[df["station_code"] == station_code]
        if station_df.empty:
            print(f"⚠️  No data for station {station_code}, skipping...")
            continue

        station_name = station_df["station_name"].iloc[0]
        print(f"\n📈 Processing station: {station_code} - {station_name}")

        # Extract time series data
        time_series, date_types = extract_time_series_data(df, station_code)

        if time_series.empty:
            print(f"⚠️  No time series data for station {station_code}, skipping...")
            continue

        print(f"   Found {len(time_series)} days with data")

        # Compute distance ratios
        results = compute_distance_ratios(time_series, date_types)

        if np.isnan(results["ratio"]):
            print(f"⚠️  Could not compute ratio for station {station_code}, skipping...")
            continue

        print(f"   Ratio R = {results['ratio']:.3f}")
        print(f"   Within-group mean: {results['within_mean']:.2f}")
        print(f"   Between-group mean: {results['between_mean']:.2f}")
        print(
            f"   Within pairs: {results['n_within']}, Between pairs: {results['n_between']}"
        )

        # Store results
        results_list.append(
            {
                "station_code": station_code,
                "station_name": station_name,
                "ratio": results["ratio"],
                "within_mean": results["within_mean"],
                "between_mean": results["between_mean"],
                "within_std": np.std(results["within_distances"])
                if results["within_distances"]
                else np.nan,
                "between_std": np.std(results["between_distances"])
                if results["between_distances"]
                else np.nan,
                "n_within": results["n_within"],
                "n_between": results["n_between"],
                "n_days": len(time_series),
            }
        )

        # Plot if requested
        if plot:
            plot_distance_distributions(
                results,
                station_code,
                station_name,
                output_dir=output_dir,
                count_type=count_type,
            )

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    if not results_df.empty:
        print("\n📊 Summary Statistics:")
        print(f"   Mean ratio: {results_df['ratio'].mean():.3f}")
        print(f"   Median ratio: {results_df['ratio'].median():.3f}")
        print(f"   Min ratio: {results_df['ratio'].min():.3f} (best separation)")
        print(f"   Max ratio: {results_df['ratio'].max():.3f} (worst separation)")

        # Save results to CSV
        csv_path = output_dir / f"distance_ratios_{count_type}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"💾 Saved results to {csv_path}")

    print(f"\n✅ Completed analysis for {len(results_df)} stations")
    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute within- vs between-group distance ratios per station"
    )
    parser.add_argument(
        "--count_type",
        choices=["checkins", "checkouts"],
        default="checkins",
        help="Type of counts to analyze (default: checkins)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/distance_results",
        help="Directory to save plots and results (default: src/workflow/distance_results)",
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
        help="Optional list of station codes to analyze (default: all from persistence)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Don't generate plots",
    )

    args = parser.parse_args()

    results_df = run_distance_analysis(
        count_type=args.count_type,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
        plot=not args.no_plot,
    )

    print("\n📊 Final Results:")
    print(results_df.to_string(index=False))
