"""
Time Rescaling Theorem QQ-Plots

This script:
1. Calculates intensity per station and day type from mean profiles
2. Computes compensators from intensity
3. Applies time rescaling theorem to raw CSV data
4. Generates QQ-plots per station and day type

The time rescaling theorem states that if we have the correct intensity,
the rescaled inter-event times should be exponentially distributed,
which transforms to uniform(0,1) via u_i = 1 - exp(-τ_i).

NOTE: This script only supports checkins, not checkouts. Checkouts data has
a minimum granularity of 15 minutes and does not contain raw timestamps per
event, which are required for the time rescaling theorem analysis.
"""

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.day_type import get_day_type
from src.workflow.data_loader import load_data, load_persisted_data
from src.workflow.data_reader import load_csv_file as load_csv_file_from_reader

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]


def sample_dates_by_daytype(
    dates: list[str], percentage: float, seed: int = 42
) -> list[str]:
    """
    Sample a percentage of dates per day type.

    Args:
        dates: List of date strings in YYYYMMDD format
        percentage: Percentage of dates to sample per day type (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of sampled date strings
    """
    if percentage >= 1.0:
        return dates

    random.seed(seed)

    # Group dates by day type
    dates_by_daytype = defaultdict(list)
    for date_str in dates:
        date_obj = datetime.strptime(date_str, "%Y%m%d").date()
        day_type = get_day_type(date_obj)
        dates_by_daytype[day_type].append(date_str)

    # Sample from each day type
    sampled_dates = []
    for day_type in DATE_TYPE_ORDER:
        if day_type not in dates_by_daytype:
            continue

        daytype_dates = dates_by_daytype[day_type]
        n_to_sample = max(1, int(len(daytype_dates) * percentage))
        sampled = random.sample(daytype_dates, min(n_to_sample, len(daytype_dates)))
        sampled_dates.extend(sampled)

    return sorted(sampled_dates)


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


def compute_mean_profiles_per_station_daytype(
    df: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute mean profiles per station and day type.

    Args:
        df: DataFrame from data_loader with columns: year, month, day, date_type,
            station_code, and time window columns (t_400, t_415, ..., t_2300)

    Returns:
        Dictionary: {station_code: {day_type: mean_profile_array}}
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

            # Fill NaN with 0
            time_series = time_series.fillna(0)

            # Check if all values are zero
            if time_series.sum().sum() == 0:
                continue

            # Compute mean profile
            mean_profile = time_series.mean(axis=0).values

            # Ensure it's in the correct order
            if len(mean_profile) != len(time_cols):
                continue

            results[station_code][day_type] = mean_profile

    return results


def compute_intensity_from_mean_profile(
    mean_profile: np.ndarray, time_window_minutes: float = 15.0
) -> np.ndarray:
    """
    Convert mean profile to intensity.

    Args:
        mean_profile: Mean counts per time window
        time_window_minutes: Length of each time window in minutes (default: 15)

    Returns:
        Intensity array (counts per minute)
    """
    return mean_profile / time_window_minutes


def compute_compensator(
    intensity: np.ndarray, time_cols: list[str], time_window_minutes: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute compensator (integrated intensity) from piecewise constant intensity.

    Args:
        intensity: Intensity array (counts per minute)
        time_cols: List of time column names (sorted)
        time_window_minutes: Length of each time window in minutes (default: 15)

    Returns:
        Tuple of (time_points, compensator_values) where:
            - time_points: Array of time points in hours
            - compensator_values: Cumulative compensator values
    """
    # Convert time columns to hours
    time_points = np.array([time_column_to_hours(col) for col in time_cols])

    # Convert time window from minutes to hours (for extending time_points)
    time_window_hours = time_window_minutes / 60.0

    # Compute compensator: Λ(t) = ∫₀ᵗ λ(s) ds
    # Intensity is in counts per minute, so we integrate over minutes
    # For piecewise constant intensity, this is cumulative sum
    compensator = np.cumsum(intensity * time_window_minutes)

    # Prepend 0 at the start
    compensator = np.concatenate([[0], compensator])

    # Time points: start of each bin, plus end of last bin
    time_points_extended = np.concatenate(
        [time_points, [time_points[-1] + time_window_hours]]
    )

    return time_points_extended, compensator


def evaluate_compensator(
    times: np.ndarray, time_points: np.ndarray, compensator_values: np.ndarray
) -> np.ndarray:
    """
    Evaluate compensator at given times using piecewise constant intensity.

    For piecewise constant intensity λ(t) constant on [t_i, t_{i+1}),
    the compensator is: Λ(t) = Σ_{j=0}^{i-1} λ_j * Δt_j + λ_i * (t - t_i)

    Args:
        times: Array of times in hours to evaluate at
        time_points: Time points where compensator is defined (bin boundaries)
        compensator_values: Compensator values at time_points

    Returns:
        Array of compensator values at times
    """
    # Clip times to valid range
    times = np.clip(times, time_points[0], time_points[-1])

    # Find which bin each time falls into
    # searchsorted with side='right' gives the index of the first point >= time
    bin_indices = np.searchsorted(time_points, times, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(time_points) - 2)

    result = np.zeros_like(times)

    for i, t in enumerate(times):
        bin_idx = bin_indices[i]
        t_start = time_points[bin_idx]
        t_end = time_points[bin_idx + 1]

        # Compensator at start of bin
        lambda_start = compensator_values[bin_idx]

        # If at the start of the bin, use that value
        if t <= t_start:
            result[i] = lambda_start
        # If at or beyond the end of the bin, use the end value
        elif t >= t_end:
            result[i] = compensator_values[bin_idx + 1]
        else:
            # Interpolate within the bin
            # For piecewise constant intensity, compensator increases linearly
            bin_width = t_end - t_start
            if bin_width > 0:
                # Intensity in this bin (constant)
                intensity = (
                    compensator_values[bin_idx + 1] - compensator_values[bin_idx]
                ) / bin_width
                # Compensator at time t
                result[i] = compensator_values[bin_idx] + intensity * (t - t_start)
            else:
                result[i] = lambda_start

    return result


def load_csv_file(
    csv_path: Path,
    station_codes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV file and extract transaction times for checkins.

    Optimized version that:
    - Uses parse_dates for faster datetime parsing
    - Filters by stations early using string pattern matching (faster)
    - Only extracts station info for filtered rows

    Args:
        csv_path: Path to CSV file
        station_codes: Optional list of station codes to filter by

    Returns:
        DataFrame with columns: datetime, station_code, station_name, hour
    """
    return load_csv_file_from_reader(
        csv_path=csv_path,
        station_codes=station_codes,
        count_type="checkins",
        include_time_components=False,
    )


def apply_time_rescaling(
    times: np.ndarray,
    time_points: np.ndarray,
    compensator_values: np.ndarray,
) -> np.ndarray:
    """
    Apply time rescaling theorem to event times.

    Args:
        times: Array of event times in hours
        time_points: Time points where compensator is defined
        compensator_values: Compensator values at time_points

    Returns:
        Array of uniform(0,1) values: u_i = 1 - exp(-τ_i)
        where τ_i = Λ(t_i) - Λ(t_{i-1})
    """
    if len(times) == 0:
        return np.array([])

    # Sort times
    sorted_times = np.sort(times)

    # Evaluate compensator at each time
    lambda_t = evaluate_compensator(sorted_times, time_points, compensator_values)

    # Compute rescaled inter-event times: τ_i = Λ(t_i) - Λ(t_{i-1})
    tau = np.diff(lambda_t, prepend=0)

    # Transform to uniform: u_i = 1 - exp(-τ_i)
    u = 1 - np.exp(-tau)

    return u


def plot_qq_plot(
    uniform_values: np.ndarray,
    station_code: str,
    station_name: str,
    day_type: str,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
    show_plots: bool = False,
) -> None:
    """
    Plot QQ-plot of uniform values against theoretical uniform(0,1).

    Args:
        uniform_values: Array of uniform(0,1) values
        station_code: Station code
        station_name: Station name
        day_type: Day type (WD, SA, SU, HO)
        output_dir: Optional directory to save plot
        count_type: Type of count ("checkins" or "checkouts")
        show_plots: Whether to display plots
    """
    if len(uniform_values) == 0:
        print(f"⚠️  No data to plot for {station_code}/{day_type}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Theoretical quantiles
    n = len(uniform_values)
    theoretical = np.linspace(0, 1, n + 2)[1:-1]  # Avoid 0 and 1

    # Sample quantiles (sorted)
    sample = np.sort(uniform_values)

    # Plot QQ-plot
    ax.scatter(theoretical, sample, alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="y=x (perfect fit)")

    # Add KS test statistic
    ks_stat, ks_pvalue = stats.kstest(uniform_values, "uniform")
    ax.text(
        0.05,
        0.95,
        f"KS statistic: {ks_stat:.4f}\np-value: {ks_pvalue:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Theoretical Quantiles (Uniform)", fontsize=12)
    ax.set_ylabel("Sample Quantiles", fontsize=12)
    ax.set_title(
        f"QQ-Plot: {station_code} - {station_name}\n"
        f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type})\n"
        f"n={len(uniform_values)}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"qq_plot_{station_code}_{day_type}_{count_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved QQ-plot to {filename}")

    if show_plots:
        plt.show()
    plt.close(fig)


def run_time_rescaling_analysis(
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    data_dir: Optional[Path] = None,
    time_window_minutes: float = 15.0,
    date_percentage: Optional[float] = None,
    date_type: Optional[list[str]] = None,
    show_plots: bool = False,
) -> dict:
    """
    Run time rescaling theorem analysis for checkins.

    NOTE: This function only supports checkins. Checkouts are not supported because
    checkout data has a minimum granularity of 15 minutes and does not contain
    raw timestamps per event, which are required for the time rescaling theorem.

    Args:
        output_dir: Directory to save plots (default: src/workflow/results/time_rescaling)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all from data.
        data_dir: Directory containing CSV files (default: data/check_ins/daily)
        time_window_minutes: Length of time window in minutes (default: 15)
        date_percentage: Optional percentage of dates to use per day type (0.0 to 1.0).
            If None, uses all dates. Uses seed from params.json for reproducibility.
        date_type: Optional list of day types to analyze (e.g., ["WD", "SA"]).
            Valid values: WD (Weekday), SA (Saturday), SU (Sunday), HO (Holiday).
            If None, uses all day types.
        show_plots: Whether to display plots

    Returns:
        Dictionary with results for each station and day type
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

    # Set data directory
    if data_dir is None:
        data_dir = Path("data/check_ins/daily")
    else:
        data_dir = Path(data_dir)

    # Load sampled dates and stations
    persistence_dir = params_path.parent
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)

    if not sampled_dates:
        raise ValueError("No sampled dates found. Run workflow steps 1-4 first.")

    # Determine which day types to analyze
    day_types_to_analyze = DATE_TYPE_ORDER
    if date_type is not None:
        # Validate provided day types
        invalid = [dt for dt in date_type if dt not in DATE_TYPE_ORDER]
        if invalid:
            raise ValueError(
                f"Invalid date_type values: {invalid}. "
                f"Valid values are: {DATE_TYPE_ORDER}"
            )
        day_types_to_analyze = [dt for dt in DATE_TYPE_ORDER if dt in date_type]
        labels = [DATE_TYPE_LABELS.get(dt, dt) for dt in day_types_to_analyze]
        print(f"📅 Filtering to day types: {', '.join(labels)}")

    # Filter sampled dates to only include the requested day types
    if date_type is not None:
        original_count = len(sampled_dates)
        sampled_dates = [
            d for d in sampled_dates
            if get_day_type(datetime.strptime(d, "%Y%m%d").date()) in day_types_to_analyze
        ]
        print(f"   {len(sampled_dates)}/{original_count} dates match selected day types")

    # Sample dates by day type if percentage is specified
    seed = params.get("seed", 42)
    if date_percentage is not None:
        original_count = len(sampled_dates)
        sampled_dates = sample_dates_by_daytype(
            sampled_dates, percentage=date_percentage, seed=seed
        )
        print(
            f"📅 Date sampling: {len(sampled_dates)}/{original_count} dates selected ({date_percentage * 100:.1f}% per day type)"
        )

        # Print breakdown by day type
        dates_by_daytype = defaultdict(list)
        for date_str in sampled_dates:
            date_obj = datetime.strptime(date_str, "%Y%m%d").date()
            day_type = get_day_type(date_obj)
            dates_by_daytype[day_type].append(date_str)

        for day_type in day_types_to_analyze:
            if day_type in dates_by_daytype:
                count = len(dates_by_daytype[day_type])
                print(f"   {DATE_TYPE_LABELS.get(day_type, day_type)}: {count} dates")
    else:
        print(f"📅 Using all {len(sampled_dates)} dates")

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

    print("📊 Loading checkins data for intensity estimation...")
    data = load_data(
        station_codes=station_codes,
        include_checkins=True,
        include_checkouts=False,
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
    )

    if "checkins" not in data:
        raise ValueError("No checkins data available")

    df = data["checkins"]

    if df.empty:
        raise ValueError("Empty DataFrame for checkins")

    print("🔍 Computing mean profiles per station and day type...")
    mean_profiles = compute_mean_profiles_per_station_daytype(df)

    # Get time columns for compensator computation
    time_cols = [col for col in df.columns if col.startswith("t_")]
    time_cols = sort_time_columns(time_cols)

    print("📈 Computing intensities and compensators...")
    intensities = {}
    compensators = {}

    for station_code, daytype_profiles in mean_profiles.items():
        intensities[station_code] = {}
        compensators[station_code] = {}

        for day_type, mean_profile in daytype_profiles.items():
            # Compute intensity
            intensity = compute_intensity_from_mean_profile(
                mean_profile, time_window_minutes=time_window_minutes
            )
            intensities[station_code][day_type] = intensity

            # Compute compensator
            time_points, compensator_values = compute_compensator(
                intensity, time_cols, time_window_minutes=time_window_minutes
            )
            compensators[station_code][day_type] = (time_points, compensator_values)

    print("📂 Processing CSV files for time rescaling...")
    print(f"   Dates: {len(sampled_dates)}")
    print(f"   Stations: {len(station_codes)}")

    # Collect uniform values per station and day type
    uniform_values = {
        station_code: {day_type: [] for day_type in DATE_TYPE_ORDER}
        for station_code in station_codes
    }

    # Process each CSV file
    for date_str in sampled_dates:
        print(f"Processing date: {date_str}")
        csv_filename = f"{date_str}.csv"
        csv_path = data_dir / csv_filename

        if not csv_path.exists():
            print(f"⚠️  File not found: {csv_path}, skipping...")
            continue

        # Parse date to get day type
        date_obj = datetime.strptime(date_str, "%Y%m%d").date()
        day_type = get_day_type(date_obj)

        # Load CSV file with station filtering (optimized - filters during load)
        try:
            csv_df = load_csv_file(csv_path, station_codes=station_codes)
        except Exception as e:
            print(f"⚠️  Error loading {csv_filename}: {e}, skipping...")
            continue

        # Process each station
        for station_code in station_codes:
            print(f"Processing station: {station_code}")
            if station_code not in compensators:
                continue

            if day_type not in compensators[station_code]:
                continue

            # Filter to this station
            station_df = csv_df[csv_df["station_code"] == station_code].copy()

            if station_df.empty:
                continue

            # Get times in hours
            times = station_df["hour"].values

            # Filter times to valid range (time_min to time_max in hours)
            time_min_hours = time_min // 100 + (time_min % 100) / 60.0
            time_max_hours = time_max // 100 + (time_max % 100) / 60.0
            mask = (times >= time_min_hours) & (times <= time_max_hours)
            times = times[mask]

            if len(times) == 0:
                continue

            # Apply time rescaling
            time_points, compensator_values = compensators[station_code][day_type]
            u_values = apply_time_rescaling(times, time_points, compensator_values)

            # Collect uniform values
            uniform_values[station_code][day_type].extend(u_values.tolist())

    print("📊 Generating QQ-plots...")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/results/time_rescaling")
    output_dir = Path(output_dir)

    results = {}
    ks_stats_rows = []

    # Generate QQ-plots
    for station_code in station_codes:
        if station_code not in uniform_values:
            continue

        # Get station name from DataFrame
        station_df = df[df["station_code"] == station_code]
        if station_df.empty:
            continue
        station_name = station_df["station_name"].iloc[0]

        results[station_code] = {}

        for day_type in day_types_to_analyze:
            if day_type not in uniform_values[station_code]:
                continue

            u_vals = np.array(uniform_values[station_code][day_type])

            if len(u_vals) == 0:
                continue

            print(f"   Plotting QQ-plot: {station_code} - {day_type} (n={len(u_vals)})")

            # Plot QQ-plot
            plot_qq_plot(
                u_vals,
                station_code,
                station_name,
                day_type,
                output_dir=output_dir,
                count_type="checkins",
                show_plots=show_plots,
            )

            # Compute KS test
            ks_stat, ks_pvalue = stats.kstest(u_vals, "uniform")

            results[station_code][day_type] = {
                "n_events": len(u_vals),
                "uniform_values": u_vals.tolist(),
            }

            ks_stats_rows.append(
                {
                    "station_code": station_code,
                    "station_name": station_name,
                    "day_type": day_type,
                    "n_events": len(u_vals),
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                }
            )

    # Save KS test statistics CSV
    if ks_stats_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        ks_df = pd.DataFrame(ks_stats_rows)
        ks_csv = output_dir / "time_rescaling_ks_stats_checkins.csv"
        ks_df.to_csv(ks_csv, index=False)
        print(f"💾 Saved KS test statistics to {ks_csv}")

    print(f"\n✅ Completed analysis for {len(results)} stations")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply time rescaling theorem and generate QQ-plots for checkins. "
        "NOTE: Checkouts are not supported due to 15-minute granularity and lack of raw timestamps."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/time_rescaling",
        help="Directory to save plots (default: src/workflow/results/time_rescaling)",
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
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing CSV files (default: data/check_ins/daily)",
    )
    parser.add_argument(
        "--time_window_minutes",
        type=float,
        default=15.0,
        help="Length of time window in minutes (default: 15)",
    )
    parser.add_argument(
        "--date_percentage",
        type=float,
        default=None,
        help="Percentage of dates to use per day type (0.0 to 1.0). If None, uses all dates. (default: None)",
    )
    parser.add_argument(
        "--date_type",
        type=str,
        nargs="+",
        choices=["WD", "SA", "SU", "HO"],
        default=None,
        help="Day types to analyze: WD (Weekday), SA (Saturday), SU (Sunday), HO (Holiday). "
        "If not specified, all day types are analyzed. (default: all)",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display the plots interactively in a window (blocks execution)",
    )

    args = parser.parse_args()

    results = run_time_rescaling_analysis(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        params_path=Path(args.params),
        station_codes=args.stations,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        time_window_minutes=args.time_window_minutes,
        date_percentage=args.date_percentage,
        date_type=args.date_type,
        show_plots=args.show_plots,
    )

    print(f"\n📊 Generated QQ-plots for {len(results)} stations")
