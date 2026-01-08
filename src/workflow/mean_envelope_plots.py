"""
Mean ± Envelope Plots per Station

This script creates mean arrival curves with envelope bands for each day type.
For each station, it plots:
- Mean arrival curve for each day type (WD, SA, SU, HO)
- Envelope showing either ±1 std band or 10-90% quantile range
- All day types overlaid on the same plot

This helps visualize the typical pattern and variability for each day type.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Order for plotting (to ensure consistent legend order)
DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]


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
    time_cols = sort_time_columns(time_cols)  # Sort by numeric time value

    # Extract time series matrix (each row is a day, columns are time windows)
    time_series = station_df[time_cols].copy()

    # Extract date types
    date_types = station_df["date_type"].copy()

    return time_series, date_types


def compute_mean_and_envelope(
    time_series: pd.DataFrame,
    date_types: pd.Series,
    envelope_type: Literal["std", "quantile"] = "std",
    quantile_low: float = 0.1,
    quantile_high: float = 0.9,
) -> tuple[dict, list[str]]:
    """
    Compute mean curves and envelopes for each day type.

    Args:
        time_series: DataFrame with rows as days and columns as time windows
        date_types: Series with date_type for each day
        envelope_type: "std" for ±1 std band, "quantile" for quantile envelope
        quantile_low: Lower quantile (default: 0.1 for 10%)
        quantile_high: Upper quantile (default: 0.9 for 90%)

    Returns:
        Tuple of (results_dict, time_columns) where:
            - results_dict: Dictionary with keys for each day type, each containing:
                - mean: Mean curve (array)
                - lower: Lower envelope (array)
                - upper: Upper envelope (array)
                - n_days: Number of days in this group
            - time_columns: List of time column names used (sorted)
    """
    results = {}
    unique_types = date_types.unique()

    # Get and sort time columns once (by numeric value, not string)
    time_cols = [c for c in time_series.columns if c.startswith("t_")]
    time_cols = sort_time_columns(time_cols)

    for day_type in unique_types:
        mask = date_types == day_type
        group_data = time_series[mask].copy()

        if group_data.empty:
            continue

        # Use the pre-sorted time columns
        if not time_cols:
            continue

        # Select columns in the correct sorted order
        # This ensures the DataFrame columns match our sorted time_cols
        group_data = group_data[time_cols].copy()

        # Verify column order matches (pandas should preserve the order we specify)
        if list(group_data.columns) != time_cols:
            # Force reorder if needed
            group_data = group_data.reindex(columns=time_cols)

        # Check if we have any non-NaN data
        if group_data.isna().all().all():
            continue

        # Check for NaN values before filling
        nan_count = group_data.isna().sum().sum()
        if nan_count > 0:
            print(f"      {day_type}: Filling {nan_count} NaN values with 0")

        # Fill NaN with 0 (or could use forward fill, but 0 is reasonable for counts)
        group_data = group_data.fillna(0)

        # Check if all values are zero (might indicate no data)
        total_sum = group_data.sum().sum()
        if total_sum == 0:
            print(
                f"      Warning: All values are 0 for {day_type} (total={total_sum}), skipping..."
            )
            continue

        # Compute mean - values will be in the same order as time_cols (sorted)
        mean_curve = group_data.mean(axis=0).values
        mean_sum = mean_curve.sum()
        print(
            f"      {day_type}: Mean curve sum = {mean_sum:.2f}, max = {mean_curve.max():.2f}, first_col={time_cols[0]}, last_col={time_cols[-1]}"
        )

        # Check if mean curve is valid
        if np.isnan(mean_curve).any() or len(mean_curve) == 0:
            print(f"      Warning: Invalid mean curve for {day_type}")
            continue

        # Compute envelope
        if envelope_type == "std":
            std_curve = group_data.std(axis=0).values
            # Handle case where std is 0 (all values same)
            lower = mean_curve - std_curve
            upper = mean_curve + std_curve
        else:  # quantile
            lower = group_data.quantile(quantile_low, axis=0).values
            upper = group_data.quantile(quantile_high, axis=0).values

        results[day_type] = {
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


def plot_mean_envelope(
    results: dict,
    time_cols: list[str],
    station_code: str,
    station_name: str,
    envelope_type: str = "std",
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot mean curves with envelopes for each day type.

    Args:
        results: Dictionary from compute_mean_and_envelope
        time_cols: List of time column names
        station_code: Station code
        station_name: Station name
        envelope_type: Type of envelope used ("std" or "quantile")
        output_dir: Optional directory to save the plot
        count_type: Type of counts ("checkins" or "checkouts")
    """
    if not results:
        print(f"⚠️  No data to plot for station {station_code}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Ensure time columns are sorted numerically (not string-sorted)
    time_cols = sort_time_columns(time_cols)

    # Convert time columns to hours
    hours = time_columns_to_hours(time_cols)

    # Debug: Verify dimensions and ordering
    print(f"   Plotting: {len(time_cols)} time columns, {len(hours)} hours")
    print(f"   Time columns: first={time_cols[0]}, last={time_cols[-1]}")
    print(f"   Hours: first={hours[0]:.2f}, last={hours[-1]:.2f}")

    # Plot each day type
    for day_type in DATE_TYPE_ORDER:
        if day_type not in results:
            continue

        data = results[day_type]
        mean = data["mean"]
        lower = data["lower"]
        upper = data["upper"]
        n_days = data["n_days"]

        # Debug: Check array shapes and values
        print(
            f"      {day_type}: mean shape={mean.shape}, len={len(mean)}, sum={mean.sum():.2f}, max={mean.max():.2f}"
        )
        print(
            f"      {day_type}: hours len={len(hours)}, first={hours[0]:.2f}, last={hours[-1]:.2f}, time_cols first={time_cols[0]}, last={time_cols[-1]}"
        )

        # Verify shapes match
        if len(mean) != len(hours):
            print(f"      ERROR: Shape mismatch! mean={len(mean)}, hours={len(hours)}")
            continue

        # Debug: Check if mean has any non-zero values
        if mean.sum() == 0:
            print(
                f"      Warning: Mean curve for {day_type} sums to 0, max={mean.max()}"
            )
            # Still plot it so user can see there's data but it's all zero

        color = DATE_TYPE_COLORS.get(day_type, "gray")
        label = f"{DATE_TYPE_LABELS.get(day_type, day_type)} (n={n_days})"

        # Plot envelope (filled area)
        ax.fill_between(
            hours,
            lower,
            upper,
            alpha=0.2,
            color=color,
            label=f"{label} envelope",
        )

        # Plot mean curve
        ax.plot(
            hours,
            mean,
            color=color,
            linewidth=2.5,
            label=label,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    envelope_label = "±1 std" if envelope_type == "std" else "10-90% quantile"
    ax.set_title(
        f"Mean ± Envelope: {station_code} - {station_name}\n"
        f"({count_type.capitalize()}, {envelope_label})",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(hours[0], hours[-1])

    # Ensure y-axis shows the data (don't let it auto-scale to 0 if there's data)
    if results:
        all_means = [data["mean"] for data in results.values()]
        all_uppers = [data["upper"] for data in results.values()]
        if all_means:
            max_val = max([m.max() for m in all_means if len(m) > 0])
            max_upper = max([u.max() for u in all_uppers if len(u) > 0])
            y_max = max(max_val, max_upper) * 1.1  # Add 10% padding
            y_min = 0
            ax.set_ylim(y_min, y_max)
            print(f"   Y-axis limits: {y_min} to {y_max:.2f}")

    # Format x-axis to show hours nicely
    ax.set_xticks(range(int(hours[0]), int(hours[-1]) + 1, 2))
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in range(int(hours[0]), int(hours[-1]) + 1, 2)]
    )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"mean_envelope_{station_code}_{count_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved plot to {filename}")

    plt.show()


def run_mean_envelope_analysis(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    envelope_type: Literal["std", "quantile"] = "std",
    quantile_low: float = 0.1,
    quantile_high: float = 0.9,
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
) -> dict:
    """
    Run mean envelope analysis for each station.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        envelope_type: "std" for ±1 std band, "quantile" for quantile envelope
        quantile_low: Lower quantile (default: 0.1 for 10%)
        quantile_high: Upper quantile (default: 0.9 for 90%)
        output_dir: Directory to save plots (default: src/workflow/envelope_results)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all from persistence.

    Returns:
        Dictionary with results for each station
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
        output_dir = Path("src/workflow/envelope_results")
    output_dir = Path(output_dir)

    results = {}

    # Get time columns (will be sorted properly in extract_time_series_data)
    # This is just for reference, actual sorting happens in the functions

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

        # Debug: Check data values
        time_cols_debug = [c for c in time_series.columns if c.startswith("t_")]
        if time_cols_debug:
            sample_data = time_series[time_cols_debug].iloc[0]
            non_zero_count = (time_series[time_cols_debug] != 0).any(axis=1).sum()
            total_sum = time_series[time_cols_debug].sum().sum()
            print(f"   Time columns: {len(time_cols_debug)}")
            print(f"   Rows with non-zero data: {non_zero_count} / {len(time_series)}")
            print(f"   Total sum of all counts: {total_sum}")
            print(f"   Sample row sum: {sample_data.sum()}")
            print(f"   Date types: {sorted(date_types.unique())}")

        # Compute mean and envelope
        station_results, computed_time_cols = compute_mean_and_envelope(
            time_series,
            date_types,
            envelope_type=envelope_type,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )

        if not station_results:
            print(
                f"⚠️  No day type groups found for station {station_code}, skipping..."
            )
            continue

        # Print summary
        for day_type, data in station_results.items():
            print(
                f"   {DATE_TYPE_LABELS.get(day_type, day_type)}: {data['n_days']} days"
            )

        # Store results
        results[station_code] = {
            "station_name": station_name,
            "results": station_results,
        }

        # Plot using the time columns from computation (ensures they match)
        # computed_time_cols should already be sorted numerically
        plot_mean_envelope(
            station_results,
            computed_time_cols,
            station_code,
            station_name,
            envelope_type=envelope_type,
            output_dir=output_dir,
            count_type=count_type,
        )

    print(f"\n✅ Completed analysis for {len(results)} stations")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create mean ± envelope plots per station for each day type"
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
        default="src/workflow/envelope_results",
        help="Directory to save plots (default: src/workflow/envelope_results)",
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

    args = parser.parse_args()

    results = run_mean_envelope_analysis(
        count_type=args.count_type,
        envelope_type=args.envelope_type,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
    )

    print(f"\n📊 Generated plots for {len(results)} stations")
