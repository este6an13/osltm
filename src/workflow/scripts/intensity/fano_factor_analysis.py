"""
Fano Factor Analysis

This script computes Fano factors across days, per bin, for each station and day type.

For fixed station (s), day type (g), and bin (k), across dates (d ∈ g):
- μ_{s,g,k} = E[N_{s,d,k}] (mean)
- σ²_{s,g,k} = Var(N_{s,d,k}) (variance)
- Fano_{s,g,k} = σ²/μ

Poisson → Fano ≈ 1
Overdispersion → Fano > 1

Generates plots showing:
- For each day type: median Fano factor across stations vs time bins
- Horizontal reference line at 1
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.workflow.data_loader import load_data, load_persisted_data

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]


def sort_time_columns(time_cols: list[str]) -> list[str]:
    """Sort time column names by their numeric time value."""

    def get_time_value(col: str) -> int:
        time_str = col.replace("t_", "")
        return int(time_str)

    return sorted(time_cols, key=get_time_value)


def time_column_to_hours(time_col: str) -> float:
    """Convert time column name (e.g., 't_400') to hours of day."""
    time_str = time_col.replace("t_", "")
    time_int = int(time_str)
    hour = time_int // 100
    minute = time_int % 100
    return hour + minute / 60.0


def compute_fano_factors(
    df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute Fano factors per station, day type, and time bin.

    Args:
        df: DataFrame from data_loader with columns: year, month, day, date_type,
            station_code, and time window columns (t_400, t_415, ..., t_2300)

    Returns:
        Dictionary: {station_code: {day_type: {time_col: fano_factor}}}
    """
    # Get time columns and sort them
    time_cols = [col for col in df.columns if col.startswith("t_")]
    time_cols = sort_time_columns(time_cols)

    if not time_cols:
        return {}

    results = {}

    # Group by station and day type
    for station_code in df["station_code"].unique():
        station_df = df[df["station_code"] == station_code].copy()
        results[station_code] = {}

        for day_type in station_df["date_type"].unique():
            daytype_df = station_df[station_df["date_type"] == day_type].copy()

            if daytype_df.empty:
                continue

            # Extract time series data
            time_series = daytype_df[time_cols].copy()

            # Fill NaN with 0 (missing counts are treated as 0)
            time_series = time_series.fillna(0)

            # Check if all values are zero
            if time_series.sum().sum() == 0:
                continue

            results[station_code][day_type] = {}

            # Compute Fano factor for each time bin
            for time_col in time_cols:
                counts = time_series[time_col].values

                # Remove any remaining NaN (shouldn't happen after fillna, but just in case)
                counts = counts[~np.isnan(counts)]

                if len(counts) == 0:
                    continue

                # Compute mean and variance
                mean = np.mean(counts)
                variance = np.var(counts, ddof=1)  # Sample variance

                # Fano factor = variance / mean
                # Handle edge case where mean is 0
                if mean > 0:
                    fano = variance / mean
                else:
                    fano = np.nan  # Cannot compute Fano factor if mean is 0

                results[station_code][day_type][time_col] = fano

    return results


def compute_median_fano_factors(
    fano_factors: dict[str, dict[str, dict[str, float]]],
    time_cols: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute median Fano factors across stations.

    Args:
        fano_factors: Dictionary from compute_fano_factors
        time_cols: List of time column names (sorted)

    Returns:
        Dictionary: {day_type: {"median": array}}
    """
    results = {}

    for day_type in DATE_TYPE_ORDER:
        # Collect Fano factors for each time bin across all stations
        fano_by_bin = {time_col: [] for time_col in time_cols}

        for station_code, daytype_factors in fano_factors.items():
            if day_type not in daytype_factors:
                continue

            for time_col in time_cols:
                if time_col in daytype_factors[day_type]:
                    fano = daytype_factors[day_type][time_col]
                    if not np.isnan(fano):
                        fano_by_bin[time_col].append(fano)

        # Compute median for each time bin
        median_values = []

        for time_col in time_cols:
            fano_list = fano_by_bin[time_col]
            if len(fano_list) > 0:
                median = np.median(fano_list)
                median_values.append(median)
            else:
                median_values.append(np.nan)

        results[day_type] = {
            "median": np.array(median_values),
        }

    return results


def plot_fano_factors(
    median_factors: dict[str, dict[str, np.ndarray]],
    time_cols: list[str],
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot Fano factors with median line.

    Args:
        median_factors: Dictionary from compute_median_fano_factors
        time_cols: List of time column names (sorted)
        output_dir: Optional directory to save plots
        count_type: Type of counts ("checkins" or "checkouts")
    """
    # Convert time columns to hours for x-axis
    time_hours = np.array([time_column_to_hours(col) for col in time_cols])

    # Create figure with subplots for each day type
    n_daytypes = len(DATE_TYPE_ORDER)
    fig, axes = plt.subplots(
        n_daytypes, 1, figsize=(12, 4 * n_daytypes), sharex=True, sharey=True
    )

    if n_daytypes == 1:
        axes = [axes]

    for idx, day_type in enumerate(DATE_TYPE_ORDER):
        ax = axes[idx]

        if day_type not in median_factors:
            ax.text(
                0.5,
                0.5,
                f"No data for {DATE_TYPE_LABELS.get(day_type, day_type)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(
                f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type.capitalize()})",
                fontsize=14,
                fontweight="bold",
            )
            continue

        median = median_factors[day_type]["median"]

        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(median)
        if not np.any(valid_mask):
            ax.text(
                0.5,
                0.5,
                f"No valid data for {DATE_TYPE_LABELS.get(day_type, day_type)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(
                f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type.capitalize()})",
                fontsize=14,
                fontweight="bold",
            )
            continue

        time_hours_valid = time_hours[valid_mask]
        median_valid = median[valid_mask]

        # Plot median line
        ax.plot(
            time_hours_valid,
            median_valid,
            "o-",
            color="blue",
            linewidth=2,
            markersize=4,
            label="Median across stations",
        )

        # Add horizontal reference line at 1 (Poisson expectation)
        ax.axhline(
            y=1.0, color="red", linestyle="--", linewidth=2, label="Poisson (Fano=1)"
        )

        ax.set_xlabel("Time of Day (hours)", fontsize=12)
        ax.set_ylabel("Fano Factor", fontsize=12)
        ax.set_title(
            f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type.capitalize()})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_hours[0] - 0.5, time_hours[-1] + 0.5)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"fano_factor_{count_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved Fano factor plot to {filename}")

    plt.show()


def run_fano_factor_analysis(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
) -> dict:
    """
    Run Fano factor analysis.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        output_dir: Directory to save plots (default: src/workflow/results/fano_factor)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all from data.

    Returns:
        Dictionary with Fano factors per station, day type, and time bin
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

    # Load sampled dates and stations
    persistence_dir = step4_params.get("persistence_dir", Path("src/workflow/data"))
    persistence_dir = Path(persistence_dir)
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)

    if not sampled_dates:
        raise ValueError("No sampled dates found. Run workflow steps 1-4 first.")

    if sampled_stations:
        available_station_codes = [s["code"] for s in sampled_stations]
        if station_codes is None:
            station_codes = available_station_codes
        else:
            # Filter to only available stations
            station_codes = [
                sc for sc in station_codes if sc in available_station_codes
            ]
    elif station_codes is None:
        raise ValueError("No station codes provided and none found in data.")

    print(f"📊 Loading {count_type} data...")
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

    print("🔍 Computing Fano factors per station, day type, and time bin...")
    print(f"   Stations: {len(station_codes)}")
    print(f"   Date-station combinations: {len(df)}")

    # Get time columns
    time_cols = [col for col in df.columns if col.startswith("t_")]
    time_cols = sort_time_columns(time_cols)

    # Compute Fano factors
    fano_factors = compute_fano_factors(df)

    print(f"✅ Computed Fano factors for {len(fano_factors)} stations")

    # Compute median across stations
    print("📈 Computing median across stations...")
    median_factors = compute_median_fano_factors(fano_factors, time_cols)

    # Generate plots
    print("📊 Generating plots...")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/results/fano_factor")
    output_dir = Path(output_dir)

    plot_fano_factors(
        median_factors,
        time_cols,
        output_dir=output_dir,
        count_type=count_type,
    )

    print(f"\n✅ Completed Fano factor analysis for {count_type}")

    return {
        "fano_factors": fano_factors,
        "median_factors": median_factors,
        "time_columns": time_cols,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Fano factors across days, per bin, per station and day type"
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
        default="src/workflow/results/fano_factor",
        help="Directory to save plots (default: src/workflow/results/fano_factor)",
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
        help="Optional list of station codes to analyze (default: all from data)",
    )

    args = parser.parse_args()

    results = run_fano_factor_analysis(
        count_type=args.count_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        params_path=Path(args.params),
        station_codes=args.stations,
    )

    print(f"\n📊 Generated Fano factor analysis for {args.count_type}")
