import os
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.db.session_v2 import SessionLocal
from src.repo.v2.counts_15min.repository import Counts15MinRepository
from src.repo.v2.processing.repository import ProcessedFileRepository
from src.repo.v2.stations.repository import StationRepository
from src.utils.day_type import get_day_type
from src.utils.stations import extract_station_info

# Example run:
# uv run python -m src.scripts.populate.v2.populate_15min_counts 20241103 20241104

PROCESS_TYPE = "daily_check_ins_counts_15min"


def compute_counts_for_station(df: pd.DataFrame, station_id: int) -> list[dict]:
    """
    Compute total counts and 1-min variance within each 15-minute window.
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


def process_counts(file_ids: Optional[List[str]] = None):
    """Main entrypoint for computing 15-minute count aggregates and variances."""
    session: Session = SessionLocal()
    counts_repo = Counts15MinRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedFileRepository(session)

    folder_path = "data/check_ins/daily"
    print("📊 Starting 15-min count computation from raw check-in files...")

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if file_ids:
        target_files = [f"{fid}.csv" for fid in file_ids]
        files_to_process = [f for f in all_files if f in target_files]
        print(f"📅 Filtering for specific dates: {', '.join(file_ids)}")
    else:
        files_to_process = all_files

    if not files_to_process:
        print("⚠️ No matching files found to process.")
        return

    # Preload all known stations
    existing_stations = {s.code: s for s in station_repo.get_all_stations()}

    for filename in files_to_process:
        file_id = filename.replace(".csv", "")

        if processed_repo.is_processed(file_id, PROCESS_TYPE):
            print(f"✅ File {filename} already processed. Skipping.")
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"📂 Processing file: {filename}")

        df = pd.read_csv(
            file_path,
            usecols=["Fecha_Transaccion", "Estacion_Parada"],
            parse_dates=["Fecha_Transaccion"],
        )

        # Gather and insert new stations if needed
        new_station_objs = []
        for station_field in df["Estacion_Parada"].unique():
            code, name = extract_station_info(station_field)
            if code and name and code not in existing_stations:
                new_station_objs.append(dict(code=code, name=name))

        if new_station_objs:
            created = station_repo.bulk_insert_stations(new_station_objs)
            existing_stations.update({s.code: s for s in created})
            print(f"🆕 Added {len(created)} new stations")

        # Compute counts + variances for each station
        all_counts = []
        for station_field, group in df.groupby("Estacion_Parada"):
            code, _ = extract_station_info(station_field)
            station = existing_stations.get(code)
            if not station:
                continue
            all_counts.extend(compute_counts_for_station(group, station.id))

        # Bulk upsert results
        if all_counts:
            counts_repo.bulk_upsert_counts(all_counts)
            print(f"💾 Inserted/updated {len(all_counts)} 15-min count records")

        processed_repo.mark_processed(file_id, PROCESS_TYPE)
        session.commit()  # One commit per file
        print(f"✅ Finished processing {filename}\n")

    print("🏁 All 15-min count computations complete.")
    session.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_ids = sys.argv[1:]
        process_counts(file_ids)
    else:
        process_counts()
