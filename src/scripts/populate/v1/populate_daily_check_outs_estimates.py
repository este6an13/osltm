import os
import sys
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


def compute_checkout_estimates(
    df: pd.DataFrame,
    station_id: int,
    estimate_repo: EstimateRepository,
):
    """Aggregate Salidas_S every 15 min, convert to 5-min lambda_out estimate, store in DB."""
    if df.empty:
        return

    # Combine date + time columns, parse to datetime, floor to 15-min
    df["timestamp"] = pd.to_datetime(
        df["Fecha_Transaccion"].astype(str) + " " + df["Tiempo"].astype(str)
    )
    df["timestamp"] = df["timestamp"].dt.floor("15min")

    # Group by 15-min intervals and sum Salidas_S
    grouped = (
        df.groupby("timestamp")["Salidas_S"]
        .sum()
        .reset_index(name="total_checkouts_15min")
    )

    # Extract temporal features
    grouped["year"] = grouped["timestamp"].dt.year
    grouped["month"] = grouped["timestamp"].dt.month
    grouped["day"] = grouped["timestamp"].dt.day
    grouped["day_of_week"] = grouped["timestamp"].dt.dayofweek
    grouped["time_int"] = (
        grouped["timestamp"].dt.hour * 100 + grouped["timestamp"].dt.minute
    )

    # Convert to 5-min window equivalent (uniform assumption: 15-min -> divide by 3)
    grouped["lambda_out"] = grouped["total_checkouts_15min"] / 3.0

    for _, row in grouped.iterrows():
        ts = row["timestamp"]
        year = int(row["year"])
        month = int(row["month"])
        day = int(row["day"])
        day_of_week = int(row["day_of_week"])
        time_int = int(row["time_int"])
        lambda_out = float(row["lambda_out"])

        date_type = get_day_type(ts)

        # Use new EstimateRepository signatures (day + date_type)
        if estimate_repo.exists(
            year=year,
            month=month,
            day=day,
            day_of_week=day_of_week,
            time=time_int,
            date_type=date_type,
            station_id=station_id,
        ):
            estimate_repo.update_estimate_lambda_out(
                year=year,
                month=month,
                day=day,
                day_of_week=day_of_week,
                time=time_int,
                date_type=date_type,
                station_id=station_id,
                lambda_out=lambda_out,
            )
        else:
            estimate_repo.create_estimate(
                year=year,
                month=month,
                day=day,
                day_of_week=day_of_week,
                time=time_int,
                date_type=date_type,
                station_id=station_id,
                estimated_lambda_out=lambda_out,
            )


def process_check_out_estimates(file_ids: Optional[List[str]] = None):
    """
    Process daily checkout CSVs and update lambda_out estimates.
    If file_ids is provided (e.g., ["20241103", "20241105"]), only those are processed.
    """
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
        session.close()
        return

    for filename in files_to_process:
        file_id = filename.replace(".csv", "")

        # Skip if already processed
        if processed_repo.is_processed(file_id, PROCESS_TYPE):
            print(f"✅ File {filename} already processed. Skipping.")
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"📂 Processing file: {filename}")

        # Read CSV; don't parse dates here because we combine Fecha_Transaccion + Tiempo
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

        # Aggregate by station code
        for station_field, group in df.groupby("Estacion"):
            code, name = extract_station_info(station_field)
            if not code or not name:
                continue

            # Ensure station exists
            station = station_repo.get_station_by_code(code)
            if not station:
                try:
                    station = station_repo.create_station(code, name)
                    print(f"🆕 Created station {code} - {name}")
                except Exception as e:
                    print(f"⚠️ Could not create station {code}: {e}")
                    continue

            # Compute and store λ_out estimates
            compute_checkout_estimates(group, station.id, estimate_repo)

        # Mark file processed (flattened model)
        processed_repo.mark_processed(file_id, PROCESS_TYPE)
        print(f"✅ Finished processing {filename}\n")

    print("🏁 All checkout estimate computations complete.")
    session.close()


if __name__ == "__main__":
    # Example usage:
    #   uv run python -m src.utils.process_check_out_estimates 20241103 20241105
    #   uv run python -m src.utils.process_check_out_estimates
    if len(sys.argv) > 1:
        process_check_out_estimates(sys.argv[1:])
    else:
        process_check_out_estimates()
