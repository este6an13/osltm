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

# uv run python -m src.scripts.populate.v1.populate_daily_check_outs_estimates 20241103 20241104

PROCESS_TYPE = "daily_check_outs_estimates"


def compute_estimates_for_station(df: pd.DataFrame, station_id: int) -> list[dict]:
    """Compute λ̂_out for each 15-min window and return as list of dicts ready for bulk insert."""
    if df.empty:
        return []

    # Combine date + time into full timestamp
    df["timestamp"] = pd.to_datetime(
        df["Fecha_Transaccion"].astype(str) + " " + df["Tiempo"].astype(str)
    )

    # Floor to 15-min intervals
    df["timestamp"] = df["timestamp"].dt.floor("15min")

    # Aggregate total checkouts per 15-min window
    grouped = (
        df.groupby("timestamp")["Salidas_S"]
        .sum()
        .reset_index(name="total_checkouts_15min")
    )

    # Temporal features
    grouped["year"] = grouped["timestamp"].dt.year
    grouped["month"] = grouped["timestamp"].dt.month
    grouped["day"] = grouped["timestamp"].dt.day
    grouped["day_of_week"] = grouped["timestamp"].dt.dayofweek
    grouped["time_int"] = (
        grouped["timestamp"].dt.hour * 100 + grouped["timestamp"].dt.minute
    )

    # Convert 15-min total → equivalent 5-min lambda_out (divide by 3)
    grouped["lambda_out"] = grouped["total_checkouts_15min"] / 3.0

    # Prepare for bulk insert/upsert
    estimate_rows = []
    for _, row in grouped.iterrows():
        timestamp = row["timestamp"]
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
                estimated_lambda_out=float(row["lambda_out"]),
            )
        )

    return estimate_rows


def process_estimates(file_ids: Optional[List[str]] = None):
    """Main entrypoint for computing lambda_out estimates using batch inserts."""
    session: Session = SessionLocal()
    estimate_repo = EstimateRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedFileRepository(session)

    folder_path = "data/check_outs/daily"
    print("📊 Starting lambda_out estimation from raw check-out files...")

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

        # Read CSV
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

        # --- Compute estimates for each station ---
        all_estimates = []
        for station_field, group in df.groupby("Estacion"):
            code, _ = extract_station_info(station_field)
            station = existing_stations.get(code)
            if not station:
                continue
            all_estimates.extend(compute_estimates_for_station(group, station.id))

        # --- Bulk upsert estimates ---
        if all_estimates:
            estimate_repo.bulk_upsert_estimates(all_estimates)
            print(f"💾 Inserted/updated {len(all_estimates)} λ_out estimates")

        # Mark file processed and commit
        processed_repo.mark_processed(file_id, PROCESS_TYPE)
        session.commit()
        print(f"✅ Finished processing {filename}\n")

    print("🏁 All λ_out estimate computations complete.")
    session.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_ids = sys.argv[1:]
        process_estimates(file_ids)
    else:
        process_estimates()
