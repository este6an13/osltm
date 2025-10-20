import os

import pandas as pd
from sqlalchemy.orm import Session

from src.db.session import SessionLocal
from src.repo.check_ins.repository import CheckInRepository
from src.repo.processing.repository import (
    ProcessedDailyCheckInsFileRepository,
)
from src.repo.stations.repository import StationRepository


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


def process_daily_check_ins():
    """
    Process all unprocessed daily check-in CSV files.
    """
    folder_path = "data/check_ins/daily"
    session: Session = SessionLocal()

    # Initialize repositories
    processed_repo = ProcessedDailyCheckInsFileRepository(session)
    station_repo = StationRepository(session)
    check_in_repo = CheckInRepository(session)

    # Iterate over files
    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        file_id = filename.replace(".csv", "")

        # Check if already processed
        if processed_repo.is_processed(file_id, "check_ins"):
            print(f"✅ File {filename} already processed. Skipping.")
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"📂 Processing file: {filename}")

        # Read only required columns
        df = pd.read_csv(
            file_path,
            usecols=["Fecha_Transaccion", "Estacion_Parada", "Numero_Tarjeta"],
            parse_dates=["Fecha_Transaccion"],
        )

        for _, row in df.iterrows():
            _timestamp = row["Fecha_Transaccion"]
            station_field = row["Estacion_Parada"]
            card_id = row["Numero_Tarjeta"]

            code, name = extract_station_info(station_field)
            if not code or not name:
                print(f"⚠️  Skipping row with invalid station format: {station_field}")
                continue

            # Get or create station
            station = station_repo.get_station_by_code(code)
            if not station:
                try:
                    station = station_repo.create_station(code, name)
                    print(f"🆕 Created station {code} - {name}")
                except Exception as e:
                    print(f"⚠️  Could not create station {code}: {e}")
                    continue

            # Create check-in
            try:
                check_in_repo.create_check_in(
                    station_id=station.id,
                    card_id=card_id,
                    timestamp=_timestamp,
                )
            except Exception as e:
                print(f"⚠️  Failed to create check-in for station {code}: {e}")

        # Mark file as processed
        processed_repo.mark_processed(file_id, "check_ins")

        print(f"✅ Finished processing file: {filename}\n")

    session.close()


if __name__ == "__main__":
    process_daily_check_ins()
