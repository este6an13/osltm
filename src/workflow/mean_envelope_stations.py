"""
Mean ± Envelope Plots Across Stations

This script creates mean arrival curves with envelope bands for each station, grouped by day type.
For each day type, it plots:
- Mean arrival curve for each station
- Envelope showing either ±1 std band or 10-90% quantile range
- All stations overlaid on the same plot, faceted by day type

This helps visualize how different stations compare within each day type.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.workflow.data_loader import load_data

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

# Order for plotting (to ensure consistent facet order)
DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]


def extract_time_series_by_daytype(
    df: pd.DataFrame, day_type: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract time series data for a specific day type.

    Args:
        df: DataFrame from data_loader with columns: year, month, day, date_type,
            station_code, station_name, and time window columns (t_400, t_415, ..., t_2300)
        day_type: Day type to filter by (WD, SA, SU, HO)

    Returns:
        Tuple of (time_series_matrix, station_codes, station_names) where:
            - time_series_matrix: DataFrame with rows as (station, day) and columns as time windows
            - station_codes: Series with station_code for each row
            - station_names: Series with station_name for each row
    """
    # Filter by day type
    daytype_df = df[df["date_type"] == day_type].copy()

    if daytype_df.empty:
        return pd.DataFrame(), pd.Series(dtype=str), pd.Series(dtype=str)

    # Get time window columns (all columns starting with 't_')
    time_cols = [col for col in daytype_df.columns if col.startswith("t_")]
    time_cols = sort_time_columns(time_cols)  # Sort by numeric time value

    # Extract time series matrix (each row is a (station, day), columns are time windows)
    time_series = daytype_df[time_cols].copy()

    # Extract station information
    station_codes = daytype_df["station_code"].copy()
    station_names = daytype_df["station_name"].copy()

    return time_series, station_codes, station_names


def compute_mean_and_envelope_by_station(
    time_series: pd.DataFrame,
    station_codes: pd.Series,
    envelope_type: Literal["std", "quantile"] = "std",
    quantile_low: float = 0.1,
    quantile_high: float = 0.9,
) -> tuple[dict, list[str]]:
    """
    Compute mean curves and envelopes for each station.

    Args:
        time_series: DataFrame with rows as (station, day) and columns as time windows
        station_codes: Series with station_code for each row
        envelope_type: "std" for ±1 std band, "quantile" for quantile envelope
        quantile_low: Lower quantile (default: 0.1 for 10%)
        quantile_high: Upper quantile (default: 0.9 for 90%)

    Returns:
        Tuple of (results_dict, time_columns) where:
            - results_dict: Dictionary with keys for each station code, each containing:
                - mean: Mean curve (array)
                - lower: Lower envelope (array)
                - upper: Upper envelope (array)
                - n_days: Number of days for this station
            - time_columns: List of time column names used (sorted)
    """
    results = {}
    unique_stations = station_codes.unique()

    # Get and sort time columns once (by numeric value, not string)
    time_cols = [c for c in time_series.columns if c.startswith("t_")]
    time_cols = sort_time_columns(time_cols)

    for station_code in unique_stations:
        mask = station_codes == station_code
        group_data = time_series[mask].copy()

        if group_data.empty:
            continue

        # Use the pre-sorted time columns
        if not time_cols:
            continue

        # Select columns in the correct sorted order
        group_data = group_data[time_cols].copy()

        # Verify column order matches
        if list(group_data.columns) != time_cols:
            group_data = group_data.reindex(columns=time_cols)

        # Check if we have any non-NaN data
        if group_data.isna().all().all():
            continue

        # Check for NaN values before filling
        nan_count = group_data.isna().sum().sum()
        if nan_count > 0:
            print(f"      {station_code}: Filling {nan_count} NaN values with 0")

        # Fill NaN with 0
        group_data = group_data.fillna(0)

        # Check if all values are zero
        total_sum = group_data.sum().sum()
        if total_sum == 0:
            print(
                f"      Warning: All values are 0 for station {station_code}, skipping..."
            )
            continue

        # Compute mean
        mean_curve = group_data.mean(axis=0).values

        # Check if mean curve is valid
        if np.isnan(mean_curve).any() or len(mean_curve) == 0:
            print(f"      Warning: Invalid mean curve for station {station_code}")
            continue

        # Compute envelope
        if envelope_type == "std":
            std_curve = group_data.std(axis=0).values
            lower = mean_curve - std_curve
            upper = mean_curve + std_curve
        else:  # quantile
            lower = group_data.quantile(quantile_low, axis=0).values
            upper = group_data.quantile(quantile_high, axis=0).values

        results[station_code] = {
            "mean": mean_curve,
            "lower": lower,
            "upper": upper,
            "n_days": len(group_data),
        }

    return results, time_cols


def time_columns_to_hours(time_cols: list[str]) -> np.ndarray:
    """
    Convert time column names (e.g., "t_400", "t_415") to hours of day.

    Args:
        time_cols: List of time column names (should be sorted by numeric time value)

    Returns:
        Array of hours (e.g., 4.0, 4.25, 4.5, ...)
    """
    hours = []
    for col in time_cols:
        # Extract time from "t_400" -> 400
        time_str = col.replace("t_", "")
        time_int = int(time_str)
        hour = time_int // 100
        minute = time_int % 100
        hours.append(hour + minute / 60.0)
    return np.array(hours)


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


def plot_mean_envelope_by_station(
    all_results: dict,
    time_cols: list[str],
    envelope_type: str = "std",
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
) -> None:
    """
    Plot mean curves with envelopes for each station, faceted by day type.

    Args:
        all_results: Dictionary with keys for each day type, each containing a dict
                     of station results (from compute_mean_and_envelope_by_station)
        time_cols: List of time column names
        envelope_type: Type of envelope used ("std" or "quantile")
        output_dir: Optional directory to save the plot
        count_type: Type of counts ("checkins" or "checkouts")
        station_codes: Optional list of station codes to include (for consistent colors)
        day_types: Optional list of day types to include (WD, SA, SU, HO). If None, uses all.
    """
    if not all_results:
        print("⚠️  No data to plot")
        return

    # Ensure time columns are sorted numerically
    time_cols = sort_time_columns(time_cols)

    # Convert time columns to hours
    hours = time_columns_to_hours(time_cols)

    # Filter day types if provided (before computing station colors)
    if day_types is not None:
        day_type_order = [dt for dt in DATE_TYPE_ORDER if dt in day_types]
        # Filter all_results to only include requested day types
        all_results = {
            dt: all_results[dt] for dt in day_type_order if dt in all_results
        }
    else:
        day_type_order = [dt for dt in DATE_TYPE_ORDER if dt in all_results]

    if not day_type_order:
        print("⚠️  No day types to plot")
        return

    # Get all unique stations across all day types for consistent coloring
    all_station_codes = set()
    for day_type_results in all_results.values():
        all_station_codes.update(day_type_results.keys())

    if station_codes is not None:
        # Filter to only requested stations
        all_station_codes = [s for s in station_codes if s in all_station_codes]
    else:
        all_station_codes = sorted(all_station_codes)

    n_stations = len(all_station_codes)

    # Assign colors to stations
    if n_stations <= 10:
        colors = sns.color_palette("tab10", n_stations)
    else:
        colors = sns.color_palette("husl", n_stations)

    station_color_map = dict(zip(all_station_codes, colors))

    # Create faceted plot using matplotlib subplots
    n_facets = len(day_type_order)
    n_cols = 2 if n_facets > 1 else 1
    n_rows = (n_facets + 1) // 2 if n_facets > 1 else 1
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 6 * n_rows), sharex=True, sharey=True
    )
    if n_facets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each day type
    for idx, day_type in enumerate(day_type_order):
        if day_type not in all_results:
            continue

        ax = axes[idx]
        day_type_results = all_results[day_type]

        # Plot each station
        for station_code in sorted(day_type_results.keys()):
            if station_code not in station_color_map:
                continue

            data = day_type_results[station_code]
            mean = data["mean"]
            lower = data["lower"]
            upper = data["upper"]
            n_days = data["n_days"]

            color = station_color_map[station_code]
            label = f"{station_code} (n={n_days})"

            # Verify shapes match
            if len(mean) != len(hours):
                print(
                    f"      ERROR: Shape mismatch for {day_type}/{station_code}! "
                    f"mean={len(mean)}, hours={len(hours)}"
                )
                continue

            # Plot envelope (filled area)
            ax.fill_between(
                hours,
                lower,
                upper,
                alpha=0.2,
                color=color,
            )

            # Plot mean curve
            ax.plot(
                hours,
                mean,
                color=color,
                linewidth=2,
                label=label,
                marker="o",
                markersize=3,
            )

        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(
            DATE_TYPE_LABELS.get(day_type, day_type), fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(hours[0], hours[-1])

        # Set y-axis limits
        if day_type_results:
            all_means = [data["mean"] for data in day_type_results.values()]
            all_uppers = [data["upper"] for data in day_type_results.values()]
            if all_means:
                max_val = max([m.max() for m in all_means if len(m) > 0])
                max_upper = max([u.max() for u in all_uppers if len(u) > 0])
                y_max = max(max_val, max_upper) * 1.1
                y_min = 0
                ax.set_ylim(y_min, y_max)

        # Format x-axis
        ax.set_xticks(range(int(hours[0]), int(hours[-1]) + 1, 2))
        ax.set_xticklabels(
            [f"{h:02d}:00" for h in range(int(hours[0]), int(hours[-1]) + 1, 2)]
        )

        # Add legend if not too many stations
        if n_stations <= 15:
            ax.legend(
                loc="upper right",
                fontsize=8,
                ncol=1 if n_stations <= 10 else 2,
                framealpha=0.9,
            )

    # Hide unused subplots
    for idx in range(len(day_type_order), len(axes)):
        axes[idx].set_visible(False)

    # Add main title
    envelope_label = "±1 std" if envelope_type == "std" else "10-90% quantile"
    fig.suptitle(
        f"Mean ± Envelope Across Stations ({count_type.capitalize()})\n"
        f"Color = Station, Facet = Day Type ({envelope_label})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"mean_envelope_stations_{count_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved plot to {filename}")

    plt.show()


def run_mean_envelope_by_station(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    envelope_type: Literal["std", "quantile"] = "std",
    quantile_low: float = 0.1,
    quantile_high: float = 0.9,
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
) -> dict:
    """
    Run mean envelope analysis across stations, grouped by day type.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        envelope_type: "std" for ±1 std band, "quantile" for quantile envelope
        quantile_low: Lower quantile (default: 0.1 for 10%)
        quantile_high: Upper quantile (default: 0.9 for 90%)
        output_dir: Directory to save plots (default: src/workflow/results/envelope_results)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all stations.
        day_types: Optional list of day types to include (WD, SA, SU, HO). If None, uses all.

    Returns:
        Dictionary with results for each day type
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

    print(f"   Found {len(df)} (station, day) profiles")

    # Determine which day types to process
    if day_types is not None:
        day_types_to_process = [dt for dt in DATE_TYPE_ORDER if dt in day_types]
        print(f"🔍 Processing day types: {day_types_to_process}")
    else:
        day_types_to_process = DATE_TYPE_ORDER
        print("🔍 Processing all day types")

    # Process each day type
    all_results = {}
    time_cols = None

    for day_type in day_types_to_process:
        print(f"\n📈 Processing day type: {DATE_TYPE_LABELS.get(day_type, day_type)}")

        # Extract time series data for this day type
        time_series, station_codes_daytype, station_names_daytype = (
            extract_time_series_by_daytype(df, day_type)
        )

        if time_series.empty:
            print(f"   ⚠️  No data for {day_type}, skipping...")
            continue

        print(f"   Found {len(time_series)} (station, day) profiles")

        # Filter by station codes if provided (for this day type)
        if station_codes is not None:
            mask = station_codes_daytype.isin(station_codes)
            time_series = time_series[mask].copy()
            station_codes_daytype = station_codes_daytype[mask].copy()
            station_names_daytype = station_names_daytype[mask].copy()

        # Compute mean and envelope for each station
        day_type_results, computed_time_cols = compute_mean_and_envelope_by_station(
            time_series,
            station_codes_daytype,
            envelope_type=envelope_type,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )

        if not day_type_results:
            print(f"   ⚠️  No station groups found for {day_type}, skipping...")
            continue

        # Store time columns (should be the same for all day types)
        if time_cols is None:
            time_cols = computed_time_cols

        # Print summary
        print(f"   Found {len(day_type_results)} stations with data")
        for station_code, data in sorted(day_type_results.items()):
            print(f"      {station_code}: {data['n_days']} days")

        all_results[day_type] = day_type_results

    if not all_results:
        raise ValueError("No data found for any day type")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/results/envelope_results")
    output_dir = Path(output_dir)

    # Plot results
    plot_mean_envelope_by_station(
        all_results,
        time_cols,
        envelope_type=envelope_type,
        output_dir=output_dir,
        count_type=count_type,
        station_codes=station_codes,
        day_types=day_types,
    )

    print(f"\n✅ Completed analysis for {len(all_results)} day types")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create mean ± envelope plots across stations, grouped by day type"
    )
    parser.add_argument(
        "--count_type",
        choices=["checkins", "checkouts"],
        default="checkins",
        help="Type of counts to analyze (default: checkins)",
    )
    parser.add_argument(
        "--envelope_type",
        choices=["std", "quantile"],
        default="std",
        help="Type of envelope: 'std' for ±1 std band, 'quantile' for quantile range (default: std)",
    )
    parser.add_argument(
        "--quantile_low",
        type=float,
        default=0.1,
        help="Lower quantile for quantile envelope (default: 0.1)",
    )
    parser.add_argument(
        "--quantile_high",
        type=float,
        default=0.9,
        help="Upper quantile for quantile envelope (default: 0.9)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/envelope_results",
        help="Directory to save plots (default: src/workflow/results/envelope_results)",
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

    results = run_mean_envelope_by_station(
        count_type=args.count_type,
        envelope_type=args.envelope_type,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
        day_types=args.day_types,
    )

    print(f"\n📊 Generated plots for {len(results)} day types")
