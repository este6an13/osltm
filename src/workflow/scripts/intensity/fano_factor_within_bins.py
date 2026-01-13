"""
Fano Factor Within Bins Analysis

This script computes Fano factors within time bins using sub-bins of configurable size.

For each file, per station, within a time bin:
- Subdivide into sub-bins of size delta_minutes
- Compute: Var[N(δ)] / E[N(δ)] where δ = delta_minutes

Plots average within-bin Fano factor per station and day type as a curve.
"""

import json
from collections import defaultdict
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    import random

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


def load_csv_file(
    csv_path: Path,
    station_codes: Optional[list[str]] = None,
    count_type: str = "checkins",
) -> pd.DataFrame:
    """
    Load a CSV file and extract transaction times.

    Args:
        csv_path: Path to CSV file
        station_codes: Optional list of station codes to filter by
        count_type: Type of counts ("checkins" or "checkouts")

    Returns:
        DataFrame with columns: datetime, station_code, station_name, hour, minute, second, hour_float
        For checkouts, counts are expanded into individual events.
    """
    return load_csv_file_from_reader(
        csv_path=csv_path,
        station_codes=station_codes,
        count_type=count_type,
        include_time_components=True,
    )


def expand_database_counts_to_events(
    df: pd.DataFrame, time_cols: list[str]
) -> pd.DataFrame:
    """
    Convert database DataFrame with time columns (t_400, t_415, etc.) into individual events.

    Args:
        df: DataFrame with columns: year, month, day, station_code, station_name,
            and time columns (t_400, t_415, ..., t_2300) containing counts
        time_cols: List of time column names (e.g., ["t_400", "t_415", ...])

    Returns:
        DataFrame with columns: datetime, station_code, station_name, hour, minute, second, hour_float
        Each row represents one event.
    """
    expanded_rows = []

    for _, row in df.iterrows():
        # Create date from year, month, day
        date_obj = datetime(int(row["year"]), int(row["month"]), int(row["day"])).date()

        station_code = row["station_code"]
        station_name = row["station_name"]

        # Process each time column
        for time_col in time_cols:
            count = row.get(time_col, 0)
            if pd.isna(count) or count == 0:
                continue

            # Extract time from column name (e.g., "t_400" -> 400)
            time_str = time_col.replace("t_", "")
            time_int = int(time_str)
            hours = time_int // 100
            minutes = time_int % 100

            # Create datetime for the start of this 15-minute window
            event_datetime = datetime.combine(
                date_obj, dt_time(hour=hours, minute=minutes, second=0)
            )

            # Expand count into individual events
            count_int = int(count)
            for _ in range(count_int):
                expanded_rows.append(
                    {
                        "datetime": event_datetime,
                        "station_code": station_code,
                        "station_name": station_name,
                    }
                )

    if not expanded_rows:
        # Return empty dataframe with correct columns
        return pd.DataFrame(
            columns=[
                "datetime",
                "station_code",
                "station_name",
                "hour",
                "minute",
                "second",
                "hour_float",
            ]
        )

    df_events = pd.DataFrame(expanded_rows)

    # Extract time components
    df_events["hour"] = df_events["datetime"].dt.hour
    df_events["minute"] = df_events["datetime"].dt.minute
    df_events["second"] = df_events["datetime"].dt.second
    df_events["hour_float"] = (
        df_events["hour"] + df_events["minute"] / 60.0 + df_events["second"] / 3600.0
    )

    return df_events[
        [
            "datetime",
            "station_code",
            "station_name",
            "hour",
            "minute",
            "second",
            "hour_float",
        ]
    ].copy()


def compute_fano_within_bin(
    events: np.ndarray,
    bin_start_minutes: float,
    bin_duration_minutes: int = 15,
    delta_minutes: int = 1,
) -> float:
    """
    Compute Fano factor within a bin by subdividing into sub-bins of size delta_minutes.

    Args:
        events: Array of event times in minutes from start of day (can be float)
        bin_start_minutes: Start of the bin in minutes from start of day
        bin_duration_minutes: Duration of the bin in minutes (default: 15)
        delta_minutes: Size of sub-bins in minutes (default: 1)

    Returns:
        Fano factor: Var[N(δ)] / E[N(δ)] where δ = delta_minutes
    """
    # Filter events to this bin
    bin_end_minutes = bin_start_minutes + bin_duration_minutes
    events_in_bin = events[(events >= bin_start_minutes) & (events < bin_end_minutes)]

    if len(events_in_bin) == 0:
        return np.nan

    # Subdivide into sub-bins of size delta_minutes
    n_sub_bins = int(bin_duration_minutes / delta_minutes)
    counts_per_sub_bin = np.zeros(n_sub_bins, dtype=int)

    for event_minutes in events_in_bin:
        # Relative position within the bin (0 to bin_duration_minutes)
        relative_minutes = event_minutes - bin_start_minutes
        # Which sub-bin does this fall into?
        sub_bin_idx = int(np.floor(relative_minutes / delta_minutes))
        if 0 <= sub_bin_idx < n_sub_bins:
            counts_per_sub_bin[sub_bin_idx] += 1

    # Compute mean and variance of counts across sub-bins
    mean = np.mean(counts_per_sub_bin)
    variance = np.var(counts_per_sub_bin, ddof=1)  # Sample variance

    # Fano factor = variance / mean
    if mean > 0:
        return variance / mean
    else:
        return np.nan


def compute_fano_factors_within_bins(
    df: pd.DataFrame,
    time_min: int = 400,
    time_max: int = 2300,
    time_step: int = 15,
    delta_minutes: int = 1,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """
    Compute Fano factors within bins for each station, day type, and time bin.

    Args:
        df: DataFrame with columns: datetime, station_code, hour, minute, hour_float
        time_min: Minimum time in HHMM format (default: 400 for 04:00)
        time_max: Maximum time in HHMM format (default: 2300 for 23:00)
        time_step: Time step in minutes for bin size (default: 15)
        delta_minutes: Size of sub-bins in minutes for Fano factor computation (default: 1)

    Returns:
        Dictionary: {station_code: {day_type: {time_col: [fano_factors]}}}
        where fano_factors is a list of Fano factors computed for each date
    """
    # Convert times to minutes from start of day (including seconds for precision)
    df["minutes_from_start"] = df["hour"] * 60 + df["minute"] + df["second"] / 60.0

    # Convert time_min and time_max to minutes
    time_min_minutes = (time_min // 100) * 60 + (time_min % 100)
    time_max_minutes = (time_max // 100) * 60 + (time_max % 100)

    # Generate 15-min bin boundaries
    bin_starts = []
    current_minutes = time_min_minutes
    while current_minutes < time_max_minutes:
        bin_starts.append(current_minutes)
        current_minutes += time_step

    # Generate time column names
    time_cols = []
    current_time = time_min
    while current_time <= time_max:
        time_cols.append(f"t_{current_time}")
        # Increment by time_step minutes
        hours = current_time // 100
        minutes = current_time % 100
        minutes += time_step
        if minutes >= 60:
            hours += 1
            minutes = 0
        current_time = hours * 100 + minutes

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Group by date, station, and day type
    df["date_str"] = df["datetime"].dt.strftime("%Y%m%d")
    df["date_obj"] = df["datetime"].dt.date
    df["day_type"] = df["date_obj"].apply(get_day_type)

    for (date_str, station_code, day_type), group_df in df.groupby(
        ["date_str", "station_code", "day_type"]
    ):
        events = group_df["minutes_from_start"].values

        # Filter to valid time range
        events = events[(events >= time_min_minutes) & (events < time_max_minutes)]

        if len(events) == 0:
            continue

        # Compute Fano factor for each bin
        for bin_start_minutes, time_col in zip(bin_starts, time_cols):
            fano = compute_fano_within_bin(
                events, bin_start_minutes, time_step, delta_minutes
            )
            if not np.isnan(fano):
                results[station_code][day_type][time_col].append(fano)

    return dict(results)


def compute_average_fano_factors(
    fano_factors: dict[str, dict[str, dict[str, list[float]]]],
    time_cols: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute average Fano factor across dates for each station, day type, and time bin.

    Args:
        fano_factors: Dictionary from compute_fano_factors_within_bins
        time_cols: List of time column names (sorted)

    Returns:
        Dictionary: {station_code: {day_type: {time_col: avg_fano}}}
    """
    results = {}

    for station_code, daytype_factors in fano_factors.items():
        results[station_code] = {}
        for day_type, time_factors in daytype_factors.items():
            results[station_code][day_type] = {}
            for time_col in time_cols:
                if time_col in time_factors:
                    fano_list = time_factors[time_col]
                    if len(fano_list) > 0:
                        results[station_code][day_type][time_col] = np.mean(fano_list)
                    else:
                        results[station_code][day_type][time_col] = np.nan
                else:
                    results[station_code][day_type][time_col] = np.nan

    return results


def compute_median_and_iqr_fano_factors(
    avg_fano_factors: dict[str, dict[str, dict[str, float]]],
    time_cols: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute median and interquartile range of average Fano factors across stations.

    Args:
        avg_fano_factors: Dictionary from compute_average_fano_factors
        time_cols: List of time column names (sorted)

    Returns:
        Dictionary: {day_type: {"median": array, "q25": array, "q75": array}}
    """
    results = {}

    for day_type in DATE_TYPE_ORDER:
        # Collect average Fano factors for each time bin across all stations
        fano_by_bin = {time_col: [] for time_col in time_cols}

        for station_code, daytype_factors in avg_fano_factors.items():
            if day_type not in daytype_factors:
                continue

            for time_col in time_cols:
                if time_col in daytype_factors[day_type]:
                    fano = daytype_factors[day_type][time_col]
                    if not np.isnan(fano):
                        fano_by_bin[time_col].append(fano)

        # Compute median and quartiles for each time bin
        median_values = []
        q25_values = []
        q75_values = []

        for time_col in time_cols:
            fano_list = fano_by_bin[time_col]
            if len(fano_list) > 0:
                median_values.append(np.median(fano_list))
                q25_values.append(np.percentile(fano_list, 25))
                q75_values.append(np.percentile(fano_list, 75))
            else:
                median_values.append(np.nan)
                q25_values.append(np.nan)
                q75_values.append(np.nan)

        results[day_type] = {
            "median": np.array(median_values),
            "q25": np.array(q25_values),
            "q75": np.array(q75_values),
        }

    return results


def plot_fano_factors(
    median_iqr: dict[str, dict[str, np.ndarray]],
    time_cols: list[str],
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot Fano factors with median and interquartile envelope.

    Args:
        median_iqr: Dictionary from compute_median_and_iqr_fano_factors
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

        if day_type not in median_iqr:
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

        median = median_iqr[day_type]["median"]
        q25 = median_iqr[day_type]["q25"]
        q75 = median_iqr[day_type]["q75"]

        # Plot interquartile envelope
        ax.fill_between(
            time_hours,
            q25,
            q75,
            alpha=0.3,
            color="blue",
            label="IQR (25th-75th percentile)",
        )

        # Plot median line
        ax.plot(
            time_hours,
            median,
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
        ax.set_ylabel("Fano Factor (within-bin)", fontsize=12)
        ax.set_title(
            f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type.capitalize()})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_hours[0] - 0.5, time_hours[-1] + 0.5)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"fano_factor_within_bins_{count_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved Fano factor plot to {filename}")

    plt.show()


def run_fano_factor_within_bins_analysis(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    data_dir: Optional[Path] = None,
    date_percentage: Optional[float] = None,
    delta_minutes: Optional[int] = None,
) -> dict:
    """
    Run Fano factor within bins analysis.

    Args:
        count_type: Type of counts to analyze ("checkins" or "checkouts")
        output_dir: Directory to save plots (default: src/workflow/results/fano_factor_within_bins)
        params_path: Path to params.json (default: src/workflow/params.json)
        station_codes: Optional list of station codes to analyze. If None, uses all from data.
        data_dir: Directory containing CSV files for checkins only (default: data/check_ins/daily).
            Not used for checkouts, which are loaded from database.
        date_percentage: Optional percentage of dates to use per day type (0.0 to 1.0).
            If None, uses all dates. Uses seed from params.json for reproducibility.
        delta_minutes: Size of sub-bins in minutes for Fano factor computation.
            If None, uses value from params.json or defaults to 1 minute.

    Returns:
        Dictionary with Fano factors per station, day type, and time bin

    Note:
        - For checkouts: Data is loaded from database using data_loader.load_data()
        - For checkins: Data is loaded from CSV files
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
    if delta_minutes is None:
        delta_minutes = step4_params.get("delta_minutes", 1)

    # Load sampled dates and stations
    persistence_dir = step4_params.get("persistence_dir", Path("src/workflow/data"))
    persistence_dir = Path(persistence_dir)
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)

    if not sampled_dates:
        raise ValueError("No sampled dates found. Run workflow steps 1-4 first.")

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

        for day_type in DATE_TYPE_ORDER:
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

    # Generate time columns
    time_cols = []
    current_time = time_min
    while current_time <= time_max:
        time_cols.append(f"t_{current_time}")
        # Increment by time_step minutes
        hours = current_time // 100
        minutes = current_time % 100
        minutes += time_step
        if minutes >= 60:
            hours += 1
            minutes = 0
        current_time = hours * 100 + minutes
    time_cols = sort_time_columns(time_cols)

    # Load data: use database for checkouts, CSV files for checkins
    if count_type == "checkouts":
        print(f"📊 Loading {count_type} from database...")
        print(f"   Dates: {len(sampled_dates)}")
        print(f"   Stations: {len(station_codes)}")

        # Load data from database
        data = load_data(
            dates=sampled_dates,
            station_codes=station_codes,
            include_checkins=False,
            include_checkouts=True,
            persistence_dir=persistence_dir,
            time_min=time_min,
            time_max=time_max,
            time_step=time_step,
        )

        if "checkouts" not in data or data["checkouts"].empty:
            raise ValueError("No checkout data loaded from database")

        # Convert database DataFrame to individual events
        print("🔄 Expanding counts to individual events...")
        combined_df = expand_database_counts_to_events(data["checkouts"], time_cols)
        print(f"   Total events: {len(combined_df)}")

    else:  # checkins
        print(f"📊 Processing {count_type} CSV files...")
        print(f"   Dates: {len(sampled_dates)}")
        print(f"   Stations: {len(station_codes)}")

        # Set data directory for checkins
        if data_dir is None:
            data_dir = Path("data/check_ins/daily")
        else:
            data_dir = Path(data_dir)

        # Process each CSV file
        all_dataframes = []
        for date_str in sampled_dates:
            csv_filename = f"{date_str}.csv"
            csv_path = data_dir / csv_filename

            if not csv_path.exists():
                print(f"⚠️  File not found: {csv_path}, skipping...")
                continue

            # Load CSV file with station filtering
            try:
                csv_df = load_csv_file(
                    csv_path, station_codes=station_codes, count_type=count_type
                )
                if not csv_df.empty:
                    all_dataframes.append(csv_df)
            except Exception as e:
                print(f"⚠️  Error loading {csv_filename}: {e}, skipping...")
                continue

        if not all_dataframes:
            raise ValueError("No data loaded from CSV files")

        # Combine all dataframes
        print("🔍 Computing Fano factors within bins...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"   Total events: {len(combined_df)}")

    # Compute Fano factors within bins
    fano_factors = compute_fano_factors_within_bins(
        combined_df,
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
        delta_minutes=delta_minutes,
    )

    print(f"✅ Computed Fano factors for {len(fano_factors)} stations")

    # Compute average Fano factors across dates
    print("📈 Computing average Fano factors across dates...")
    avg_fano_factors = compute_average_fano_factors(fano_factors, time_cols)

    # Compute median and IQR across stations
    print("📈 Computing median and IQR across stations...")
    median_iqr = compute_median_and_iqr_fano_factors(avg_fano_factors, time_cols)

    # Generate plots
    print("📊 Generating plots...")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/results/fano_factor_within_bins")
    output_dir = Path(output_dir)

    plot_fano_factors(
        median_iqr, time_cols, output_dir=output_dir, count_type=count_type
    )

    print(f"\n✅ Completed Fano factor within bins analysis for {count_type}")

    return {
        "fano_factors": fano_factors,
        "avg_fano_factors": avg_fano_factors,
        "median_iqr": median_iqr,
        "time_columns": time_cols,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Fano factors within bins using sub-minute counts"
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
        default="src/workflow/results/fano_factor_within_bins",
        help="Directory to save plots (default: src/workflow/results/fano_factor_within_bins)",
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
        help="Directory containing CSV files for checkins only (default: data/check_ins/daily). Not used for checkouts, which are loaded from database.",
    )
    parser.add_argument(
        "--date_percentage",
        type=float,
        default=None,
        help="Percentage of dates to use per day type (0.0 to 1.0). If None, uses all dates. (default: None)",
    )
    parser.add_argument(
        "--delta_minutes",
        type=int,
        default=None,
        help="Size of sub-bins in minutes for Fano factor computation. If None, uses value from params.json or defaults to 1 minute. (default: None)",
    )

    args = parser.parse_args()

    results = run_fano_factor_within_bins_analysis(
        count_type=args.count_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        params_path=Path(args.params),
        station_codes=args.stations,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        date_percentage=args.date_percentage,
        delta_minutes=args.delta_minutes,
    )

    print(f"\n📊 Generated Fano factor within bins analysis for {args.count_type}")
