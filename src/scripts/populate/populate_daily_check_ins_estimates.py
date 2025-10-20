import math
import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from src.db.session import SessionLocal
from src.repo.estimates.repository import EstimateRepository
from src.repo.processing.repository import ProcessedDailyCheckInsFileRepository
from src.repo.stations.repository import StationRepository


def get_week_of_month(date: datetime) -> int:
    """Return the week number within the month (1–5)."""
    first_day = date.replace(day=1)
    dom = date.day
    adjusted_dom = dom + first_day.weekday()
    return int(math.ceil(adjusted_dom / 7.0))


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


def compute_estimates_for_station(
    df: pd.DataFrame,
    station_id: int,
    estimate_repo: EstimateRepository,
    interval_minutes: int = 15,  # set to 5 or 15 as you prefer
):
    """
    Compute estimates by counting events *per interval* (non-overlapping bins).
    interval_minutes: bin width in minutes (5, 15, ...).
    """
    if df.empty:
        return

    df["timestamp"] = pd.to_datetime(df["Fecha_Transaccion"])
    df = df.sort_values("timestamp")

    if df.empty:
        return

    # Floor timestamps to the desired bin (e.g., '15T' for 15 minutes)
    freq = f"{interval_minutes}min"
    df["time_bin"] = df["timestamp"].dt.floor(freq)

    # Extract date/time features from the bin start
    df["year"] = df["time_bin"].dt.year
    df["month"] = df["time_bin"].dt.month
    df["day_of_week"] = df["time_bin"].dt.dayofweek
    df["week_of_month"] = df["time_bin"].apply(get_week_of_month)
    # convert time_bin to integer HHMM for storage (e.g., 04:15 -> 415)
    df["time_int"] = df["time_bin"].dt.hour * 100 + df["time_bin"].dt.minute

    # Count events per bin
    per_bin_counts = (
        df.groupby(["year", "month", "week_of_month", "day_of_week", "time_int"])
        .size()
        .reset_index(name="count_in_interval")
    )

    # If you want an average across multiple days in the same group
    # (e.g., same weekday/week_of_month/month), take the mean of counts:
    per_bin_avg = (
        per_bin_counts.groupby(
            ["year", "month", "week_of_month", "day_of_week", "time_int"]
        )["count_in_interval"]
        .mean()
        .reset_index(name="lambda_estimate")
    )

    # Store results (lambda_estimate is average arrivals *per interval_minutes*)
    for _, row in per_bin_avg.iterrows():
        year = int(row["year"])
        month = int(row["month"])
        week_of_month = int(row["week_of_month"])
        day_of_week = int(row["day_of_week"])
        time = int(row["time_int"])  # e.g., 400, 415, 430
        is_holiday = False
        lambda_estimate = float(row["lambda_estimate"])

        if estimate_repo.exists(
            year=year,
            month=month,
            week_of_month=week_of_month,
            day_of_week=day_of_week,
            time=time,
            is_holiday=is_holiday,
            station_id=station_id,
        ):
            estimate_repo.update_estimate_lambda_in(
                year=year,
                month=month,
                week_of_month=week_of_month,
                day_of_week=day_of_week,
                time=time,
                is_holiday=is_holiday,
                station_id=station_id,
                lambda_in=lambda_estimate,
            )
        else:
            estimate_repo.create_estimate(
                year=year,
                month=month,
                week_of_month=week_of_month,
                day_of_week=day_of_week,
                time=time,
                is_holiday=is_holiday,
                station_id=station_id,
                estimated_lambda_in=lambda_estimate,
            )


def process_estimates():
    """Main entrypoint for computing lambda estimates directly from CSVs."""
    session: Session = SessionLocal()
    estimate_repo = EstimateRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedDailyCheckInsFileRepository(session)

    folder_path = "data/check_ins/daily"
    print("📊 Starting lambda estimation from raw check-in files...")

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        file_id = filename.replace(".csv", "")

        # Skip processed
        if processed_repo.is_processed(file_id, "estimates"):
            print(f"✅ File {filename} already processed. Skipping.")
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"📂 Processing file: {filename}")

        # Load minimal columns
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
        processed_repo.mark_processed(file_id, "estimates")
        print(f"✅ Finished processing {filename}\n")

    print("🏁 All estimate computations complete.")
    session.close()


if __name__ == "__main__":
    process_estimates()
