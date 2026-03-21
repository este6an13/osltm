"""
Fano Factor Within Bins Analysis

This script computes Fano factors within time bins using sub-bins of configurable size.

For each file, per station, within a time bin:
- Subdivide into sub-bins of size delta_minutes
- Compute: Var[N(δ)] / E[N(δ)] where δ = delta_minutes

Plots average within-bin Fano factor per station and day type as a curve with:
- Median Fano factor across stations vs time bins
- Envelope showing either ±1 std band or quantile range (configurable)
- Horizontal reference line at 1
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


def compute_mean_and_envelope_per_station(
    fano_factors: dict[str, dict[str, dict[str, list[float]]]],
    time_cols: list[str],
    envelope_type: Literal["std", "quantile"] = "quantile",
    quantile_low: float = 0.25,
    quantile_high: float = 0.75,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """
    Compute mean and envelope (std or quantile) across dates for each station, day type, and time bin.

    Args:
        fano_factors: Dictionary from compute_fano_factors_within_bins
        time_cols: List of time column names (sorted)
        envelope_type: "std" for ±1 std band, "quantile" for quantile envelope
        quantile_low: Lower quantile (default: 0.25 for 25th percentile)
        quantile_high: Upper quantile (default: 0.75 for 75th percentile)

    Returns:
        Dictionary: {station_code: {day_type: {time_col: {"mean": float, "lower": float, "upper": float}}}}
    """
    results = {}

    for station_code, daytype_factors in fano_factors.items():
        results[station_code] = {}
        for day_type, time_factors in daytype_factors.items():
            results[station_code][day_type] = {}
            for time_col in time_cols:
                if time_col in time_factors:
                    fano_list = time_factors[time_col]
                    # Filter out NaN values
                    fano_list = [f for f in fano_list if not np.isnan(f)]

                    if len(fano_list) > 0:
                        mean_val = np.mean(fano_list)

                        if envelope_type == "std":
                            std_val = np.std(fano_list, ddof=1)  # Sample std
                            lower_val = mean_val - std_val
                            upper_val = mean_val + std_val
                        else:  # quantile
                            lower_val = np.percentile(fano_list, quantile_low * 100)
                            upper_val = np.percentile(fano_list, quantile_high * 100)

                        results[station_code][day_type][time_col] = {
                            "mean": mean_val,
                            "lower": lower_val,
                            "upper": upper_val,
                        }
                    else:
                        results[station_code][day_type][time_col] = {
                            "mean": np.nan,
                            "lower": np.nan,
                            "upper": np.nan,
                        }
                else:
                    results[station_code][day_type][time_col] = {
                        "mean": np.nan,
                        "lower": np.nan,
                        "upper": np.nan,
                    }

    return results


def compute_median_and_envelope_across_stations(
    station_envelopes: dict[str, dict[str, dict[str, dict[str, float]]]],
    time_cols: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Aggregate mean and envelope across stations by taking median of means and aggregating envelopes.

    Args:
        station_envelopes: Dictionary from compute_mean_and_envelope_per_station
            Format: {station_code: {day_type: {time_col: {"mean": float, "lower": float, "upper": float}}}}
        time_cols: List of time column names (sorted)

    Returns:
        Dictionary: {day_type: {"median": array, "lower": array, "upper": array}}
        where median is the median of means across stations, and lower/upper are aggregated envelopes
    """
    results = {}

    for day_type in DATE_TYPE_ORDER:
        # Collect means and envelopes for each time bin across all stations
        means_by_bin = {time_col: [] for time_col in time_cols}
        lowers_by_bin = {time_col: [] for time_col in time_cols}
        uppers_by_bin = {time_col: [] for time_col in time_cols}

        for station_code, daytype_factors in station_envelopes.items():
            if day_type not in daytype_factors:
                continue

            for time_col in time_cols:
                if time_col in daytype_factors[day_type]:
                    data = daytype_factors[day_type][time_col]
                    mean_val = data["mean"]
                    lower_val = data["lower"]
                    upper_val = data["upper"]

                    if not (
                        np.isnan(mean_val) or np.isnan(lower_val) or np.isnan(upper_val)
                    ):
                        means_by_bin[time_col].append(mean_val)
                        lowers_by_bin[time_col].append(lower_val)
                        uppers_by_bin[time_col].append(upper_val)

        # Compute median of means and aggregate envelopes for each time bin
        median_values = []
        lower_values = []
        upper_values = []

        for time_col in time_cols:
            means_list = means_by_bin[time_col]
            lowers_list = lowers_by_bin[time_col]
            uppers_list = uppers_by_bin[time_col]

            if len(means_list) > 0:
                # Median of means across stations
                median_values.append(np.median(means_list))
                # Aggregate envelopes: take median of lower bounds and median of upper bounds
                # This gives us a representative envelope across stations
                lower_values.append(np.median(lowers_list))
                upper_values.append(np.median(uppers_list))
            else:
                median_values.append(np.nan)
                lower_values.append(np.nan)
                upper_values.append(np.nan)

        results[day_type] = {
            "median": np.array(median_values),
            "lower": np.array(lower_values),
            "upper": np.array(upper_values),
        }

    return results


def plot_fano_factors(
    median_envelope: dict[str, dict[str, np.ndarray]],
    time_cols: list[str],
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
    envelope_type: Literal["std", "quantile"] = "quantile",
    quantile_low: float = 0.25,
    quantile_high: float = 0.75,
) -> None:
    """
    Plot Fano factors with median and envelope (std or quantile).

    Args:
        median_envelope: Dictionary from compute_median_and_envelope_fano_factors
        time_cols: List of time column names (sorted)
        output_dir: Optional directory to save plots
        count_type: Type of counts ("checkins" or "checkouts")
        envelope_type: Type of envelope used ("std" or "quantile")
        quantile_low: Lower quantile used (for label)
        quantile_high: Upper quantile used (for label)
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

        if day_type not in median_envelope:
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

        median = median_envelope[day_type]["median"]
        lower = median_envelope[day_type]["lower"]
        upper = median_envelope[day_type]["upper"]

        # Filter out NaN values for plotting
        valid_mask = ~(np.isnan(median) | np.isnan(lower) | np.isnan(upper))
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
        lower_valid = lower[valid_mask]
        upper_valid = upper[valid_mask]

        # Create envelope label
        if envelope_type == "std":
            envelope_label = "±1 std"
        else:
            envelope_label = (
                f"{int(quantile_low * 100)}th-{int(quantile_high * 100)}th percentile"
            )

        # Plot envelope (filled area)
        ax.fill_between(
            time_hours_valid,
            lower_valid,
            upper_valid,
            alpha=0.2,
            color="blue",
            label=f"Envelope ({envelope_label})",
            edgecolor="blue",
            linewidth=0.5,
        )

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
        ax.set_ylabel("Fano Factor (within-bin)", fontsize=12)
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
    time_step: Optional[int] = None,
    delta_minutes: Optional[int] = None,
    envelope_type: Literal["std", "quantile"] = "quantile",
    quantile_low: float = 0.25,
    quantile_high: float = 0.75,
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
        time_step: Size of outer bins in minutes for Fano factor computation.
            If None, uses value from params.json or defaults to 15 minutes.
        delta_minutes: Size of sub-bins in minutes for Fano factor computation.
            If None, uses value from params.json or defaults to 1 minute.
        envelope_type: "std" for ±1 std band, "quantile" for quantile envelope (default: quantile)
        quantile_low: Lower quantile (default: 0.25 for 25th percentile)
        quantile_high: Upper quantile (default: 0.75 for 75th percentile)

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
    if time_step is None:
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

    # Compute mean and envelope per station-day_type-bin across dates
    print("📈 Computing mean and envelope per station across dates...")
    station_envelopes = compute_mean_and_envelope_per_station(
        fano_factors,
        time_cols,
        envelope_type=envelope_type,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )

    # Aggregate across stations (median of means, aggregate envelopes)
    print("📈 Aggregating across stations...")
    median_envelope = compute_median_and_envelope_across_stations(
        station_envelopes,
        time_cols,
    )

    # Generate plots
    print("📊 Generating plots...")

    # Set output directory
    if output_dir is None:
        output_dir = Path("src/workflow/results/fano_factor_within_bins")
    output_dir = Path(output_dir)

    plot_fano_factors(
        median_envelope,
        time_cols,
        output_dir=output_dir,
        count_type=count_type,
        envelope_type=envelope_type,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )

    # Save median envelope CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    envelope_rows = []
    for day_type, data in median_envelope.items():
        for i, time_col in enumerate(time_cols):
            envelope_rows.append(
                {
                    "day_type": day_type,
                    "time_bin": time_col,
                    "median": data["median"][i],
                    "lower": data["lower"][i],
                    "upper": data["upper"][i],
                }
            )
    if envelope_rows:
        envelope_df = pd.DataFrame(envelope_rows)
        envelope_csv = output_dir / f"fano_factor_within_bins_{count_type}.csv"
        envelope_df.to_csv(envelope_csv, index=False)
        print(f"💾 Saved Fano factor within-bins data to {envelope_csv}")

    print(f"\n✅ Completed Fano factor within bins analysis for {count_type}")

    return {
        "fano_factors": fano_factors,
        "station_envelopes": station_envelopes,
        "median_envelope": median_envelope,
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
        "--time_step",
        type=int,
        default=None,
        help="Size of outer bins in minutes for Fano factor computation. If None, uses value from params.json or defaults to 15 minutes. (default: None)",
    )
    parser.add_argument(
        "--delta_minutes",
        type=int,
        default=None,
        help="Size of sub-bins in minutes for Fano factor computation. If None, uses value from params.json or defaults to 1 minute. (default: None)",
    )
    parser.add_argument(
        "--envelope_type",
        choices=["std", "quantile"],
        default="quantile",
        help="Type of envelope: 'std' for ±1 std band, 'quantile' for quantile range (default: quantile)",
    )
    parser.add_argument(
        "--quantile_low",
        type=float,
        default=0.25,
        help="Lower quantile for quantile envelope (default: 0.25)",
    )
    parser.add_argument(
        "--quantile_high",
        type=float,
        default=0.75,
        help="Upper quantile for quantile envelope (default: 0.75)",
    )

    args = parser.parse_args()

    results = run_fano_factor_within_bins_analysis(
        count_type=args.count_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        params_path=Path(args.params),
        station_codes=args.stations,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        date_percentage=args.date_percentage,
        time_step=args.time_step,
        delta_minutes=args.delta_minutes,
        envelope_type=args.envelope_type,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
    )

    print(f"\n📊 Generated Fano factor within bins analysis for {args.count_type}")
