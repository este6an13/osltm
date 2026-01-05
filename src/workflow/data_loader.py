"""
Data loader for workflow: loads and transforms database counts into pandas DataFrames.

This module provides a standardized way to load count data from the database
based on sampled dates and stations, returning DataFrames ready for analysis.

Each row in the output represents one date-station combination with time windows
as columns (e.g., t_400, t_415, ..., t_2300).
"""

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.db.session_v2 import SessionLocal
from src.repo.v2.counts_15min.repository import Counts15MinRepository


def generate_time_columns(
    time_min: int = 400, time_max: int = 2300, time_step: int = 15
) -> list[str]:
    """Generate column names for time windows (t_400, t_415, ..., t_2300)."""
    columns = []
    current = time_min
    while current <= time_max:
        columns.append(f"t_{current}")
        # Increment by time_step minutes
        hours = current // 100
        minutes = current % 100
        minutes += time_step
        if minutes >= 60:
            hours += 1
            minutes = 0
        current = hours * 100 + minutes
    return columns


def load_persisted_data(
    persistence_dir: Path,
) -> tuple[Optional[list[str]], Optional[list[dict]]]:
    """
    Load persisted sampled_dates and sampled_stations from CSV files.

    Returns:
        Tuple of (sampled_dates, sampled_stations) or (None, None) if files don't exist
    """
    dates_file = persistence_dir / "sampled_dates.csv"
    stations_file = persistence_dir / "sampled_stations.csv"

    sampled_dates = None
    sampled_stations = None

    if dates_file.exists():
        df = pd.read_csv(dates_file)
        sampled_dates = df["date"].astype(str).tolist()
        print(f"📂 Loaded {len(sampled_dates)} persisted dates from {dates_file}")

    if stations_file.exists():
        df = pd.read_csv(stations_file, dtype={"code": str})
        sampled_stations = df.to_dict("records")
        print(
            f"📂 Loaded {len(sampled_stations)} persisted stations from {stations_file}"
        )

    return sampled_dates, sampled_stations


def pivot_counts_to_dataframe(
    counts: list,
    count_type: Literal["in", "out"],
    time_columns: list[str],
) -> pd.DataFrame:
    """
    Pivot count records into a DataFrame where each row is a date-station combination.

    Args:
        counts: List of Counts15Min records
        count_type: "in" for check-ins, "out" for check-outs
        time_columns: List of time column names (e.g., ["t_400", "t_415", ...])

    Returns:
        DataFrame with columns: year, month, day, date_type, station_id, station_code,
        station_name, and time window columns (t_400, t_415, ..., t_2300)
    """
    if not counts:
        # Return empty DataFrame with correct structure
        columns = [
            "year",
            "month",
            "day",
            "date_type",
            "station_id",
            "station_code",
            "station_name",
        ] + time_columns
        return pd.DataFrame(columns=columns)

    # Build list of records
    records = []
    for count in counts:
        # Get count value based on type
        count_value = count.count_in if count_type == "in" else count.count_out
        if count_value is None:
            continue

        # Create key for this date-station combination
        key = (
            count.year,
            count.month,
            count.day,
            count.date_type,
            count.station_id,
            count.station.code,
            count.station.name,
        )

        # Initialize record if not exists
        if not any(r["key"] == key for r in records):
            records.append(
                {
                    "key": key,
                    "year": count.year,
                    "month": count.month,
                    "day": count.day,
                    "date_type": count.date_type,
                    "station_id": count.station_id,
                    "station_code": count.station.code,
                    "station_name": count.station.name,
                }
            )

        # Find the record and set the time window value
        record = next(r for r in records if r["key"] == key)
        time_col = f"t_{count.time}"
        if time_col in time_columns:
            record[time_col] = count_value

    # Convert to DataFrame
    df = pd.DataFrame(records)
    if df.empty:
        columns = [
            "year",
            "month",
            "day",
            "date_type",
            "station_id",
            "station_code",
            "station_name",
        ] + time_columns
        return pd.DataFrame(columns=columns)

    # Remove key column and ensure all time columns exist
    df = df.drop(columns=["key"], errors="ignore")
    for col in time_columns:
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    base_columns = [
        "year",
        "month",
        "day",
        "date_type",
        "station_id",
        "station_code",
        "station_name",
    ]
    df = df[base_columns + time_columns]

    # Sort by date and station
    df = df.sort_values(["year", "month", "day", "station_code"]).reset_index(drop=True)

    return df


def load_data(
    dates: Optional[list[str]] = None,
    station_codes: Optional[list[str]] = None,
    include_checkins: bool = True,
    include_checkouts: bool = True,
    persistence_dir: Optional[Path] = None,
    session: Optional[Session] = None,
    time_min: int = 400,
    time_max: int = 2300,
    time_step: int = 15,
) -> dict[str, pd.DataFrame]:
    """
    Load count data from database and return as pivoted DataFrames.

    By default, loads data based on sampled dates and stations from persistence folder.

    Args:
        dates: Optional list of date strings in YYYYMMDD format.
            If None, loads from persistence folder.
        station_codes: Optional list of station codes to filter by.
            If None, loads from persistence folder or includes all.
        include_checkins: Whether to load check-in counts (default: True)
        include_checkouts: Whether to load check-out counts (default: True)
        persistence_dir: Path to persistence directory (default: src/workflow/persistence)
        session: Optional database session. If None, creates a new one.
        time_min: Minimum time in HHMM format (default: 400 for 04:00)
        time_max: Maximum time in HHMM format (default: 2300 for 23:00)
        time_step: Time step in minutes (default: 15)

    Returns:
        Dictionary with keys:
            - "checkins": DataFrame with check-in counts (if include_checkins=True)
            - "checkouts": DataFrame with check-out counts (if include_checkouts=True)

        Each DataFrame has columns:
            - year, month, day: Date components
            - date_type: Day type (WD, SA, SU, HO)
            - station_id: Station ID
            - station_code: Station code
            - station_name: Station name
            - t_400, t_415, ..., t_2300: Count values for each 15-minute window

    Example:
        >>> data = load_data()
        >>> checkins_df = data["checkins"]
        >>> checkouts_df = data["checkouts"]
        >>> # Each row represents one date-station combination
        >>> # Columns t_400 through t_2300 contain the counts for each time window
    """
    # Default persistence directory
    if persistence_dir is None:
        persistence_dir = Path("src/workflow/persistence")

    # Load dates and stations from persistence if not provided
    if dates is None or station_codes is None:
        persisted_dates, persisted_stations = load_persisted_data(persistence_dir)
        if dates is None:
            dates = persisted_dates
        if station_codes is None and persisted_stations:
            station_codes = [s["code"] for s in persisted_stations]

    if not dates:
        raise ValueError(
            "No dates provided and none found in persistence. Run workflow steps 1-4 first."
        )

    # Convert station_codes to set if provided
    station_codes_set = set(station_codes) if station_codes else None

    # Create session if not provided
    own_session = session is None
    if own_session:
        session = SessionLocal()

    try:
        # Query database
        counts_repo = Counts15MinRepository(session)
        counts = counts_repo.get_counts_by_dates_and_stations(
            dates=dates,
            station_codes=station_codes_set,
            include_checkins=include_checkins,
            include_checkouts=include_checkouts,
        )

        print(f"📊 Loaded {len(counts)} count records from database")
        print(f"   Dates: {len(dates)}")
        print(f"   Stations: {len(station_codes_set) if station_codes_set else 'all'}")

        # Generate time columns
        time_columns = generate_time_columns(
            time_min=time_min, time_max=time_max, time_step=time_step
        )

        # Build result dictionary
        result = {}

        if include_checkins:
            checkins_df = pivot_counts_to_dataframe(counts, "in", time_columns)
            result["checkins"] = checkins_df
            print(f"   Check-ins: {len(checkins_df)} date-station combinations")

        if include_checkouts:
            checkouts_df = pivot_counts_to_dataframe(counts, "out", time_columns)
            result["checkouts"] = checkouts_df
            print(f"   Check-outs: {len(checkouts_df)} date-station combinations")

        return result

    finally:
        if own_session:
            session.close()


if __name__ == "__main__":
    # Example usage
    print("Loading data from persistence...")
    data = load_data()

    if "checkins" in data:
        print("\n📥 Check-ins DataFrame:")
        print(data["checkins"].head())
        print(f"Shape: {data['checkins'].shape}")

    if "checkouts" in data:
        print("\n📤 Check-outs DataFrame:")
        print(data["checkouts"].head())
        print(f"Shape: {data['checkouts'].shape}")
