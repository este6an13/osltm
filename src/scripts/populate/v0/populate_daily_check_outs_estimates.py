import math
import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from src.db.session_v0 import SessionLocal
from src.repo.v0.estimates.repository import EstimateRepository
from src.repo.v0.processing.repository import ProcessedDailyCheckInsFileRepository
from src.repo.v0.stations.repository import StationRepository


def get_week_of_month(date: datetime) -> int:
    """Return the week number within the month (1–5)."""
    first_day = date.replace(day=1)
    dom = date.day
    adjusted_dom = dom + first_day.weekday()
    return int(math.ceil(adjusted_dom / 7.0))


def extract_station_info(station_field: str):
    """
    Extract station code and name from the 'Estacion' column.
    Example: "(02202)Calle 127 - L Oreal Paris" -> ("02202", "Calle 127 - L Oreal Paris")
    """
    if not station_field or "(" not in station_field or ")" not in station_field:
        return None, None
    code = station_field.split(")")[0].replace("(", "").strip()
    name = station_field.split(")")[1].strip()
    return code, name


def compute_checkout_estimates(
    df: pd.DataFrame,
    station_id: int,
    estimate_repo: EstimateRepository,
):
    """Aggregate Salidas_S every 15 min, convert to 5-min lambda_out estimate."""
    if df.empty:
        return

    # Combine date + time
    df["timestamp"] = pd.to_datetime(df["Fecha_Transaccion"] + " " + df["Tiempo"])
    df["timestamp"] = df["timestamp"].dt.floor("15min")

    # Group by 15-min intervals and sum Salidas_S
    grouped = (
        df.groupby(df["timestamp"])["Salidas_S"]
        .sum()
        .reset_index(name="total_checkouts_15min")
    )

    # Extract time features
    grouped["year"] = grouped["timestamp"].dt.year
    grouped["month"] = grouped["timestamp"].dt.month
    grouped["day_of_week"] = grouped["timestamp"].dt.dayofweek
    grouped["week_of_month"] = grouped["timestamp"].apply(get_week_of_month)
    grouped["time_int"] = (
        grouped["timestamp"].dt.hour * 100 + grouped["timestamp"].dt.minute
    )

    # Convert to 5-min window equivalent (uniform assumption)
    grouped["lambda_out"] = grouped["total_checkouts_15min"] / 3.0

    for _, row in grouped.iterrows():
        year = int(row["year"])
        month = int(row["month"])
        week_of_month = int(row["week_of_month"])
        day_of_week = int(row["day_of_week"])
        time = int(row["time_int"])
        lambda_out = float(row["lambda_out"])
        is_holiday = False

        # Check if record exists
        if estimate_repo.exists(
            year=year,
            month=month,
            week_of_month=week_of_month,
            day_of_week=day_of_week,
            time=time,
            is_holiday=is_holiday,
            station_id=station_id,
        ):
            # Update existing record
            estimate_repo.update_estimate_lambda_out(
                year=year,
                month=month,
                week_of_month=week_of_month,
                day_of_week=day_of_week,
                time=time,
                is_holiday=is_holiday,
                station_id=station_id,
                lambda_out=lambda_out,
            )
        else:
            # Create new
            estimate_repo.create_estimate(
                year=year,
                month=month,
                week_of_month=week_of_month,
                day_of_week=day_of_week,
                time=time,
                is_holiday=is_holiday,
                station_id=station_id,
                estimated_lambda_out=lambda_out,
            )


def process_check_out_estimates():
    """Process daily checkout CSVs and update lambda_out estimates."""
    session: Session = SessionLocal()
    estimate_repo = EstimateRepository(session)
    station_repo = StationRepository(session)
    processed_repo = ProcessedDailyCheckInsFileRepository(
        session
    )  # can reuse same repo

    folder_path = "data/check_outs/daily"
    print("📊 Starting lambda_out estimation from raw check-out files...")

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        file_id = filename.replace(".csv", "")

        # Skip processed
        if processed_repo.is_processed(file_id, "checkout_estimates"):
            print(f"✅ File {filename} already processed. Skipping.")
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"📂 Processing file: {filename}")

        df = pd.read_csv(
            file_path,
            usecols=["Fecha_Transaccion", "Tiempo", "Estacion", "Salidas_S"],
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

        # Mark file processed
        processed_repo.mark_processed(file_id, "checkout_estimates")
        print(f"✅ Finished processing {filename}\n")

    print("🏁 All checkout estimate computations complete.")
    session.close()


if __name__ == "__main__":
    process_check_out_estimates()
