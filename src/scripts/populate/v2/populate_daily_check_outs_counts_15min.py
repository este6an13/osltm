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
# uv run python -m src.scripts.populate.v2.populate_15min_checkouts 20241103 20241104

PROCESS_TYPE = "daily_check_outs_counts_15min"


def compute_counts_for_station(df: pd.DataFrame, station_id: int) -> list[dict]:
    """
    Compute total check-out counts per 15-minute window for a single station.
    Returns a list of dicts ready for bulk upsert into counts_15min.
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


def process_counts(file_ids: Optional[List[str]] = None):
    """Main entrypoint for computing 15-minute checkout counts (no variance)."""
    session: Session = SessionLocal()
    counts_repo = Counts15MinRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedFileRepository(session)

    folder_path = "data/check_outs/daily"
    print("📊 Starting 15-min checkout count aggregation from raw files...")

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

        # Load the raw checkout file
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

        # --- Gather stations ---
        new_station_objs = []
        for station_field in df["Estacion"].unique():
            code, name = extract_station_info(station_field)
            if code and name and code not in existing_stations:
                new_station_objs.append(dict(code=code, name=name))

        # --- Bulk insert new stations ---
        if new_station_objs:
            created = station_repo.bulk_insert_stations(new_station_objs)
            existing_stations.update({s.code: s for s in created})
            print(f"🆕 Added {len(created)} new stations")

        # --- Compute checkout counts ---
        all_counts = []
        for station_field, group in df.groupby("Estacion"):
            code, _ = extract_station_info(station_field)
            station = existing_stations.get(code)
            if not station:
                continue
            all_counts.extend(compute_counts_for_station(group, station.id))

        # --- Bulk upsert counts (count_out only) ---
        if all_counts:
            counts_repo.bulk_upsert_counts(all_counts)
            print(f"💾 Inserted/updated {len(all_counts)} checkout count records")

        processed_repo.mark_processed(file_id, PROCESS_TYPE)
        session.commit()
        print(f"✅ Finished processing {filename}\n")

    print("🏁 All checkout count computations complete.")
    session.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_ids = sys.argv[1:]
        process_counts(file_ids)
    else:
        process_counts()
