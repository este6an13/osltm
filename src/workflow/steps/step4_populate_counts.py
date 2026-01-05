"""
Step 4: Populate database with 15-minute counts for sampled dates and stations.

This step:
1. Processes check-in and check-out files for sampled dates
2. Filters by sampled stations only
3. Stores only counts between 400 and 2300 (04:00 to 23:00)
4. Saves progress to CSV files for persistence/resume capability
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.db.session_v2 import SessionLocal
from src.repo.v2.counts_15min.repository import Counts15MinRepository
from src.repo.v2.processing.repository import ProcessedFileRepository
from src.repo.v2.stations.repository import StationRepository
from src.utils.day_type import get_day_type
from src.utils.stations import extract_station_info
from src.workflow.data_loader import load_persisted_data

PROCESS_TYPE_INS = "daily_check_ins_counts_15min"
PROCESS_TYPE_OUTS = "daily_check_outs_counts_15min"


def save_persisted_data(
    persistence_dir: Path, sampled_dates: list[str], sampled_stations: list[dict]
) -> None:
    """Save sampled_dates and sampled_stations to CSV files for persistence."""
    persistence_dir.mkdir(parents=True, exist_ok=True)

    # Save dates
    dates_df = pd.DataFrame({"date": sampled_dates})
    dates_file = persistence_dir / "sampled_dates.csv"
    dates_df.to_csv(dates_file, index=False)
    print(f"💾 Saved {len(sampled_dates)} dates to {dates_file}")

    # Save stations
    stations_df = pd.DataFrame(sampled_stations)
    stations_file = persistence_dir / "sampled_stations.csv"
    stations_df.to_csv(stations_file, index=False)
    print(f"💾 Saved {len(sampled_stations)} stations to {stations_file}")


def compute_checkin_counts_for_station(
    df: pd.DataFrame, station_id: int, time_min: int = 400, time_max: int = 2300
) -> list[dict]:
    """
    Compute total check-in counts and 1-min variance within each 15-minute window.
    Filters to time range specified by time_min and time_max.
    Returns a list of dicts ready for bulk upsert.
    """
    if df.empty:
        return []

    # Ensure timestamps are parsed and sorted
    df["timestamp"] = pd.to_datetime(df["Fecha_Transaccion"])
    df = df.sort_values("timestamp")

    # Compute per-minute counts
    df["window_1min"] = df["timestamp"].dt.floor("1min")
    per_min_counts = df.groupby("window_1min").size().reset_index(name="count_1min")

    # Compute 15-min window each 1-min sample belongs to
    per_min_counts["window_15min"] = per_min_counts["window_1min"].dt.floor("15min")

    # Aggregate to 15-min totals + variance (computed on 1-min counts)
    per_15min_stats = (
        per_min_counts.groupby("window_15min")["count_1min"]
        .agg(["sum", "var"])
        .reset_index()
        .rename(columns={"sum": "count_in", "var": "variance_in_1min"})
    )

    # Replace NaN variances (when only 1 sample) with 0
    per_15min_stats["variance_in_1min"] = per_15min_stats["variance_in_1min"].fillna(
        0.0
    )

    # Add time breakdown columns
    per_15min_stats["year"] = per_15min_stats["window_15min"].dt.year
    per_15min_stats["month"] = per_15min_stats["window_15min"].dt.month
    per_15min_stats["day"] = per_15min_stats["window_15min"].dt.day
    per_15min_stats["day_of_week"] = per_15min_stats["window_15min"].dt.dayofweek
    per_15min_stats["time_int"] = (
        per_15min_stats["window_15min"].dt.hour * 100
        + per_15min_stats["window_15min"].dt.minute
    )

    # Filter by time range
    per_15min_stats = per_15min_stats[
        (per_15min_stats["time_int"] >= time_min)
        & (per_15min_stats["time_int"] <= time_max)
    ]

    # Prepare dicts for bulk upsert
    count_rows = []
    for _, row in per_15min_stats.iterrows():
        timestamp = row["window_15min"]
        date_type = get_day_type(timestamp)
        count_rows.append(
            dict(
                year=int(row["year"]),
                month=int(row["month"]),
                day=int(row["day"]),
                day_of_week=int(row["day_of_week"]),
                time=int(row["time_int"]),
                date_type=date_type,
                station_id=station_id,
                count_in=int(row["count_in"]),
                variance_in_1min=float(row["variance_in_1min"]),
            )
        )

    return count_rows


def compute_checkout_counts_for_station(
    df: pd.DataFrame, station_id: int, time_min: int = 400, time_max: int = 2300
) -> list[dict]:
    """
    Compute total check-out counts per 15-minute window for a single station.
    Filters to time range specified by time_min and time_max.
    Returns a list of dicts ready for bulk upsert.
    """
    if df.empty:
        return []

    # Combine date + time into a single timestamp
    df["timestamp"] = pd.to_datetime(
        df["Fecha_Transaccion"].astype(str) + " " + df["Tiempo"].astype(str)
    )

    # Floor timestamps to 15-minute windows
    df["window_15min"] = df["timestamp"].dt.floor("15min")

    # Aggregate total checkouts per 15-min interval across all machines
    grouped = (
        df.groupby("window_15min")["Salidas_S"].sum().reset_index(name="count_out")
    )

    # Temporal feature extraction
    grouped["year"] = grouped["window_15min"].dt.year
    grouped["month"] = grouped["window_15min"].dt.month
    grouped["day"] = grouped["window_15min"].dt.day
    grouped["day_of_week"] = grouped["window_15min"].dt.dayofweek
    grouped["time_int"] = (
        grouped["window_15min"].dt.hour * 100 + grouped["window_15min"].dt.minute
    )

    # Filter by time range
    grouped = grouped[
        (grouped["time_int"] >= time_min) & (grouped["time_int"] <= time_max)
    ]

    # Build rows for bulk upsert
    count_rows = []
    for _, row in grouped.iterrows():
        timestamp = row["window_15min"]
        date_type = get_day_type(timestamp)
        count_rows.append(
            dict(
                year=int(row["year"]),
                month=int(row["month"]),
                day=int(row["day"]),
                day_of_week=int(row["day_of_week"]),
                time=int(row["time_int"]),
                date_type=date_type,
                station_id=station_id,
                count_out=int(row["count_out"]),
            )
        )

    return count_rows


def process_checkins(
    session: Session,
    sampled_dates: list[str],
    sampled_station_codes: set[str],
    sampled_stations: list[dict],
    ins_path: Path,
    persistence_dir: Path,
    time_min: int = 400,
    time_max: int = 2300,
) -> None:
    """Process check-in files for sampled dates and stations."""
    counts_repo = Counts15MinRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedFileRepository(session)

    print(
        f"\n📊 Processing check-ins for {len(sampled_dates)} dates and {len(sampled_station_codes)} stations"
    )
    print(
        f"   Time range: {time_min} ({time_min // 100:02d}:00) to {time_max} ({time_max // 100:02d}:00)"
    )

    # Preload stations and ensure sampled stations exist
    existing_stations = {s.code: s for s in station_repo.get_all_stations()}

    # Create missing stations
    new_station_objs = []
    for station in sampled_stations:
        code = station["code"]
        name = station["name"]
        if code not in existing_stations:
            new_station_objs.append(dict(code=code, name=name))

    if new_station_objs:
        created = station_repo.bulk_insert_stations(new_station_objs)
        existing_stations.update({s.code: s for s in created})
        print(f"🆕 Created {len(created)} new stations in database")

    station_id_map = {
        code: existing_stations[code].id
        for code in sampled_station_codes
        if code in existing_stations
    }

    if len(station_id_map) < len(sampled_station_codes):
        missing = sampled_station_codes - set(station_id_map.keys())
        print(f"⚠️  Warning: {len(missing)} sampled stations not found in database")

    total_processed = 0
    for date_str in sampled_dates:
        file_id = date_str
        filename = f"{date_str}.csv"
        file_path = ins_path / filename

        if not file_path.exists():
            print(f"⚠️  File not found: {filename}, skipping...")
            continue

        if processed_repo.is_processed(file_id, PROCESS_TYPE_INS):
            print(f"✅ File {filename} already processed. Skipping.")
            continue

        print(f"📂 Processing check-ins: {filename}")

        try:
            df = pd.read_csv(
                file_path,
                usecols=["Fecha_Transaccion", "Estacion_Parada"],
                parse_dates=["Fecha_Transaccion"],
            )

            # Filter by sampled stations
            df["station_code"] = df["Estacion_Parada"].apply(
                lambda x: extract_station_info(x)[0]
            )
            df = df[df["station_code"].isin(sampled_station_codes)]

            if df.empty:
                print("   No data for sampled stations in this file")
                processed_repo.mark_processed(file_id, PROCESS_TYPE_INS)
                session.commit()
                continue

            # Compute counts for each sampled station
            all_counts = []
            for station_code, group in df.groupby("station_code"):
                station_id = station_id_map.get(station_code)
                if not station_id:
                    continue
                all_counts.extend(
                    compute_checkin_counts_for_station(
                        group, station_id, time_min, time_max
                    )
                )

            # Bulk upsert results
            if all_counts:
                counts_repo.bulk_upsert_counts(all_counts)
                print(
                    f"   💾 Inserted/updated {len(all_counts)} check-in count records"
                )

            processed_repo.mark_processed(file_id, PROCESS_TYPE_INS)
            session.commit()
            total_processed += 1
            print(f"   ✅ Finished processing {filename}\n")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            session.rollback()
            continue

    print(f"🏁 Processed {total_processed}/{len(sampled_dates)} check-in files")


def process_checkouts(
    session: Session,
    sampled_dates: list[str],
    sampled_station_codes: set[str],
    sampled_stations: list[dict],
    outs_path: Path,
    persistence_dir: Path,
    time_min: int = 400,
    time_max: int = 2300,
) -> None:
    """Process check-out files for sampled dates and stations."""
    counts_repo = Counts15MinRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedFileRepository(session)

    print(
        f"\n📊 Processing check-outs for {len(sampled_dates)} dates and {len(sampled_station_codes)} stations"
    )
    print(
        f"   Time range: {time_min} ({time_min // 100:02d}:00) to {time_max} ({time_max // 100:02d}:00)"
    )

    # Preload stations and ensure sampled stations exist
    existing_stations = {s.code: s for s in station_repo.get_all_stations()}

    # Create missing stations
    new_station_objs = []
    for station in sampled_stations:
        code = station["code"]
        name = station["name"]
        if code not in existing_stations:
            new_station_objs.append(dict(code=code, name=name))

    if new_station_objs:
        created = station_repo.bulk_insert_stations(new_station_objs)
        existing_stations.update({s.code: s for s in created})
        print(f"🆕 Created {len(created)} new stations in database")

    station_id_map = {
        code: existing_stations[code].id
        for code in sampled_station_codes
        if code in existing_stations
    }

    if len(station_id_map) < len(sampled_station_codes):
        missing = sampled_station_codes - set(station_id_map.keys())
        print(f"⚠️  Warning: {len(missing)} sampled stations not found in database")

    total_processed = 0
    for date_str in sampled_dates:
        file_id = date_str
        filename = f"{date_str}.csv"
        file_path = outs_path / filename

        if not file_path.exists():
            print(f"⚠️  File not found: {filename}, skipping...")
            continue

        if processed_repo.is_processed(file_id, PROCESS_TYPE_OUTS):
            print(f"✅ File {filename} already processed. Skipping.")
            continue

        print(f"📂 Processing check-outs: {filename}")

        try:
            df = pd.read_csv(
                file_path,
                usecols=["Fecha_Transaccion", "Tiempo", "Estacion", "Salidas_S"],
                dtype={
                    "Fecha_Transaccion": str,
                    "Tiempo": str,
                    "Estacion": str,
                    "Salidas_S": float,
                },
            )

            # Filter by sampled stations
            df["station_code"] = df["Estacion"].apply(
                lambda x: extract_station_info(x)[0]
            )
            df = df[df["station_code"].isin(sampled_station_codes)]

            if df.empty:
                print("   No data for sampled stations in this file")
                processed_repo.mark_processed(file_id, PROCESS_TYPE_OUTS)
                session.commit()
                continue

            # Compute counts for each sampled station
            all_counts = []
            for station_code, group in df.groupby("station_code"):
                station_id = station_id_map.get(station_code)
                if not station_id:
                    continue
                all_counts.extend(
                    compute_checkout_counts_for_station(
                        group, station_id, time_min, time_max
                    )
                )

            # Bulk upsert results
            if all_counts:
                counts_repo.bulk_upsert_counts(all_counts)
                print(
                    f"   💾 Inserted/updated {len(all_counts)} check-out count records"
                )

            processed_repo.mark_processed(file_id, PROCESS_TYPE_OUTS)
            session.commit()
            total_processed += 1
            print(f"   ✅ Finished processing {filename}\n")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            session.rollback()
            continue

    print(f"🏁 Processed {total_processed}/{len(sampled_dates)} check-out files")


def run(params: dict[str, Any]) -> None:
    """
    Execute step 4: populate database with 15-minute counts.

    Parameters are read from:
        - params['sampled_dates']: List of dates in YYYYMMDD format (from step 1)
        - params['sampled_stations']: List of station dicts with 'code' and 'name' (from step 3)
        - params['step2']: Paths to check-in/out directories
        - params['step4']: Configuration dict with:
            - persistence_dir: Directory to save/load persisted data (default: src/workflow/persistence)
            - process_checkins: Whether to process check-ins (default: true)
            - process_checkouts: Whether to process check-outs (default: true)
    """
    step4_params = params.get("step4", {})
    persistence_dir = Path(
        step4_params.get("persistence_dir", "src/workflow/persistence")
    )
    process_checkins_flag = step4_params.get("process_checkins", True)
    process_checkouts_flag = step4_params.get("process_checkouts", True)
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)

    # Try to load persisted data first
    persisted_dates, persisted_stations = load_persisted_data(persistence_dir)

    # Get dates from params or persisted data
    sampled_dates = params.get("sampled_dates") or persisted_dates
    if not sampled_dates:
        raise ValueError(
            "No sampled_dates found in params or persistence. Run step 1 first."
        )

    # Get stations from params or persisted data
    sampled_stations = params.get("sampled_stations") or persisted_stations
    if not sampled_stations:
        raise ValueError(
            "No sampled_stations found in params or persistence. Run step 3 first."
        )

    # Save to persistence if not already persisted
    if not persisted_dates or not persisted_stations:
        save_persisted_data(persistence_dir, sampled_dates, sampled_stations)

    # Extract station codes
    sampled_station_codes = {s["code"] for s in sampled_stations}

    # Get paths from step 2
    step2_params = params.get("step2", {})
    ins_path = Path(step2_params.get("ins_path", "data/check_ins/daily"))
    outs_path = Path(step2_params.get("outs_path", "data/check_outs/daily"))

    print("🗄️  Populating database with 15-minute counts")
    print(f"   Dates: {len(sampled_dates)}")
    print(f"   Stations: {len(sampled_station_codes)}")
    print(f"   Check-ins: {'Yes' if process_checkins_flag else 'No'}")
    print(f"   Check-outs: {'Yes' if process_checkouts_flag else 'No'}")

    session: Session = SessionLocal()

    try:
        if process_checkins_flag:
            process_checkins(
                session,
                sampled_dates,
                sampled_station_codes,
                sampled_stations,
                ins_path,
                persistence_dir,
                time_min,
                time_max,
            )

        if process_checkouts_flag:
            process_checkouts(
                session,
                sampled_dates,
                sampled_station_codes,
                sampled_stations,
                outs_path,
                persistence_dir,
                time_min,
                time_max,
            )

        print("\n✅ All database population complete!")

    finally:
        session.close()


if __name__ == "__main__":
    # For testing
    import json
    from pathlib import Path

    params_path = Path(__file__).parent.parent / "params.json"
    with open(params_path) as f:
        params = json.load(f)

    # Simulate previous steps if needed
    if "sampled_dates" not in params or params["sampled_dates"] is None:
        print("⚠️  No sampled_dates in params, using test dates")
        params["sampled_dates"] = ["20240625", "20240628"]

    if "sampled_stations" not in params or params["sampled_stations"] is None:
        print("⚠️  No sampled_stations in params, using test stations")
        params["sampled_stations"] = [
            {"code": "02202", "name": "Calle 127 - L Oreal Paris"},
            {"code": "04102", "name": "Avenida Boyaca"},
        ]

    run(params)
