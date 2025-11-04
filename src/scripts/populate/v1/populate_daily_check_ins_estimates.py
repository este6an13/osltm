import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from src.db.session_v1 import SessionLocal
from src.repo.v1.estimates.repository import EstimateRepository
from src.repo.v1.processing.repository import ProcessedFileRepository
from src.repo.v1.stations.repository import StationRepository
from src.utils.colombian_holidays import is_colombian_holiday


def extract_station_info(station_field: str):
    """
    Extract station code and name from the 'Estacion_Parada' column.
    Example: "(09104) Restrepo" -> ("09104", "Restrepo")
    """
    if not station_field or "(" not in station_field or ")" not in station_field:
        return None, None
    code = station_field.split(")")[0].replace("(", "").strip()
    name = station_field.split(")")[1].strip()
    return code, name


def get_date_type(date_obj: datetime) -> str:
    """
    Classify the date as 'weekday', 'saturday', 'sunday', or 'holiday' (Colombia).
    """
    if is_colombian_holiday(date_obj):
        return "H"  # holiday
    day_of_week = date_obj.weekday()
    if day_of_week == 5:
        return "SA"  # satuday
    elif day_of_week == 6:
        return "SU"  # sunday
    return "WD"  # weekday


def compute_estimates_for_station(
    df: pd.DataFrame,
    station_id: int,
    estimate_repo: EstimateRepository,
):
    """
    Compute λ̂ for each 15-min window as the average of the three 5-min counts
    within that window and store it in the database.
    """
    if df.empty:
        return

    df["timestamp"] = pd.to_datetime(df["Fecha_Transaccion"])
    df = df.sort_values("timestamp")

    # Step 1: Count arrivals per 5-minute window
    df["window_5min"] = df["timestamp"].dt.floor("5min")
    per_5min_counts = df.groupby("window_5min").size().reset_index(name="count_5min")

    # Step 2: Map each 5-min window to its parent 15-min window
    per_5min_counts["window_15min"] = per_5min_counts["window_5min"].dt.floor("15min")

    # Step 3: For each 15-min window, average the 5-min counts
    per_15min_avg = (
        per_5min_counts.groupby("window_15min")["count_5min"]
        .mean()
        .reset_index(name="lambda_estimate")
    )

    # Step 4: Extract temporal features for database storage
    per_15min_avg["year"] = per_15min_avg["window_15min"].dt.year
    per_15min_avg["month"] = per_15min_avg["window_15min"].dt.month
    per_15min_avg["day"] = per_15min_avg["window_15min"].dt.day
    per_15min_avg["day_of_week"] = per_15min_avg["window_15min"].dt.dayofweek
    per_15min_avg["time_int"] = (
        per_15min_avg["window_15min"].dt.hour * 100
        + per_15min_avg["window_15min"].dt.minute
    )

    # Step 5: Store results
    for _, row in per_15min_avg.iterrows():
        timestamp = row["window_15min"]
        year = int(row["year"])
        month = int(row["month"])
        day = int(row["day"])
        day_of_week = int(row["day_of_week"])
        time = int(row["time_int"])
        lambda_estimate = float(row["lambda_estimate"])
        date_type = get_date_type(timestamp)

        if estimate_repo.exists(
            year=year,
            month=month,
            day=day,
            day_of_week=day_of_week,
            time=time,
            date_type=date_type,
            station_id=station_id,
        ):
            estimate_repo.update_estimate_lambda_in(
                year=year,
                month=month,
                day=day,
                day_of_week=day_of_week,
                time=time,
                date_type=date_type,
                station_id=station_id,
                lambda_in=lambda_estimate,
            )
        else:
            estimate_repo.create_estimate(
                year=year,
                month=month,
                day=day,
                day_of_week=day_of_week,
                time=time,
                date_type=date_type,
                station_id=station_id,
                estimated_lambda_in=lambda_estimate,
            )


def process_estimates():
    """Main entrypoint for computing lambda estimates directly from CSVs."""
    session: Session = SessionLocal()
    estimate_repo = EstimateRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedFileRepository(session)

    folder_path = "data/check_ins/daily"
    print("📊 Starting lambda estimation from raw check-in files...")

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        file_id = filename.replace(".csv", "")

        # Skip processed files
        if processed_repo.is_processed(file_id, "daily_check_ins_estimates"):
            print(f"✅ File {filename} already processed. Skipping.")
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"📂 Processing file: {filename}")

        # Load required columns
        df = pd.read_csv(
            file_path,
            usecols=["Fecha_Transaccion", "Estacion_Parada"],
            parse_dates=["Fecha_Transaccion"],
        )

        # Process per station
        for station_field, group in df.groupby("Estacion_Parada"):
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

            # Compute λ estimates
            compute_estimates_for_station(group, station.id, estimate_repo)

        # Mark file processed
        processed_repo.mark_processed(file_id, "daily_check_ins_estimates")
        print(f"✅ Finished processing {filename}\n")

    print("🏁 All estimate computations complete.")
    session.close()


if __name__ == "__main__":
    process_estimates()
