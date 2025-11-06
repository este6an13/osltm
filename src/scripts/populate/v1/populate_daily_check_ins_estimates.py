import os
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.db.session_v1 import SessionLocal
from src.repo.v1.estimates.repository import EstimateRepository
from src.repo.v1.processing.repository import ProcessedFileRepository
from src.repo.v1.stations.repository import StationRepository
from src.utils.day_type import get_day_type
from src.utils.stations import extract_station_info

# uv run python -m src.scripts.populate.v1.populate_daily_check_ins_estimates 20241103 20241104

PROCESS_TYPE = "daily_check_ins_estimates"


def compute_estimates_for_station(df: pd.DataFrame, station_id: int) -> list[dict]:
    """Compute λ̂ for each 15-min window and return as list of dicts ready for bulk insert."""
    if df.empty:
        return []

    df["timestamp"] = pd.to_datetime(df["Fecha_Transaccion"])
    df = df.sort_values("timestamp")

    # Count arrivals per 5-minute window
    df["window_5min"] = df["timestamp"].dt.floor("5min")
    per_5min_counts = df.groupby("window_5min").size().reset_index(name="count_5min")

    # Aggregate to 15-minute average
    per_5min_counts["window_15min"] = per_5min_counts["window_5min"].dt.floor("15min")
    per_15min_avg = (
        per_5min_counts.groupby("window_15min")["count_5min"]
        .mean()
        .reset_index(name="lambda_estimate")
    )

    # Add features
    per_15min_avg["year"] = per_15min_avg["window_15min"].dt.year
    per_15min_avg["month"] = per_15min_avg["window_15min"].dt.month
    per_15min_avg["day"] = per_15min_avg["window_15min"].dt.day
    per_15min_avg["day_of_week"] = per_15min_avg["window_15min"].dt.dayofweek
    per_15min_avg["time_int"] = (
        per_15min_avg["window_15min"].dt.hour * 100
        + per_15min_avg["window_15min"].dt.minute
    )

    # Prepare for bulk insert/upsert
    estimate_rows = []
    for _, row in per_15min_avg.iterrows():
        timestamp = row["window_15min"]
        date_type = get_day_type(timestamp)
        estimate_rows.append(
            dict(
                year=int(row["year"]),
                month=int(row["month"]),
                day=int(row["day"]),
                day_of_week=int(row["day_of_week"]),
                time=int(row["time_int"]),
                date_type=date_type,
                station_id=station_id,
                estimated_lambda_in=float(row["lambda_estimate"]),
            )
        )

    return estimate_rows


def process_estimates(file_ids: Optional[List[str]] = None):
    """Main entrypoint for computing lambda estimates using batch inserts."""
    session: Session = SessionLocal()
    estimate_repo = EstimateRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedFileRepository(session)

    folder_path = "data/check_ins/daily"
    print("📊 Starting lambda estimation from raw check-in files...")

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

    # --- Preload all known stations once ---
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

        # --- Gather stations ---
        new_station_objs = []
        for station_field in df["Estacion_Parada"].unique():
            code, name = extract_station_info(station_field)
            if code and name and code not in existing_stations:
                new_station_objs.append(dict(code=code, name=name))

        # --- Bulk insert new stations ---
        if new_station_objs:
            created = station_repo.bulk_insert_stations(new_station_objs)
            existing_stations.update({s.code: s for s in created})
            print(f"🆕 Added {len(created)} new stations")

        # --- Compute estimates in memory ---
        all_estimates = []
        for station_field, group in df.groupby("Estacion_Parada"):
            code, _ = extract_station_info(station_field)
            station = existing_stations.get(code)
            if not station:
                continue
            all_estimates.extend(compute_estimates_for_station(group, station.id))

        # --- Bulk upsert estimates ---
        if all_estimates:
            estimate_repo.bulk_upsert_estimates(all_estimates)
            print(f"💾 Inserted/updated {len(all_estimates)} estimates")

        processed_repo.mark_processed(file_id, PROCESS_TYPE)
        session.commit()  # One commit per file
        print(f"✅ Finished processing {filename}\n")

    print("🏁 All estimate computations complete.")
    session.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_ids = sys.argv[1:]
        process_estimates(file_ids)
    else:
        process_estimates()
