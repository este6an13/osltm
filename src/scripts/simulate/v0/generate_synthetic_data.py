import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.constants.seed import SEED
from src.db.session_v0 import SessionLocal
from src.repo.v0.estimates.models import Estimate
from src.repo.v0.stations.repository import StationRepository


def simulate_poisson_series(lambdas: list[float], seed: int | None = None):
    """Generate Poisson-distributed counts for each λ."""
    rng = np.random.default_rng(seed)
    return rng.poisson(lambdas)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_week_of_month(date: datetime) -> int:
    """Return the week number within the month (1–5)."""
    first_day = date.replace(day=1)
    dom = date.day
    adjusted_dom = dom + first_day.weekday()
    return int(np.ceil(adjusted_dom / 7.0))


def generate_synthetic_data_for_date(
    target_date: datetime,
    mode: str = "both",  # "in", "out", or "both"
    seed: int | None = SEED,
):
    """
    Generate individual-level synthetic check-ins and/or check-outs for all stations.

    Each row represents a single simulated passenger event.
    """
    session: Session = SessionLocal()
    station_repo = StationRepository(session)

    year = target_date.year
    month = target_date.month
    week_of_month = get_week_of_month(target_date)
    day_of_week = target_date.weekday()

    rng = np.random.default_rng(seed)
    records_in, records_out = [], []

    stations = station_repo.get_all_stations()
    if not stations:
        print("⚠️ No stations found in the database.")
        return

    for station in stations:
        estimates = (
            session.query(Estimate)
            .filter(
                Estimate.station_id == station.id,
                Estimate.year == year,
                Estimate.month == month,
                Estimate.week_of_month == week_of_month,
                Estimate.day_of_week == day_of_week,
            )
            .order_by(Estimate.time.asc())
            .all()
        )

        if not estimates:
            continue

        interval_minutes = 5
        sub_intervals = 3  # each estimate covers 3 × 5min = 15min

        for e in estimates:
            base_time = datetime(
                year=year,
                month=month,
                day=target_date.day,
                hour=e.time // 100,
                minute=e.time % 100,
            )

            # --- Generate check-ins ---
            if (
                mode in ("in", "both")
                and e.estimated_lambda_in
                and e.estimated_lambda_in > 0
            ):
                for i in range(sub_intervals):
                    start = base_time + timedelta(minutes=i * interval_minutes)
                    n_in = rng.poisson(e.estimated_lambda_in)
                    offsets = rng.uniform(0, interval_minutes * 60, size=n_in)
                    for offset in offsets:
                        ts = start + timedelta(seconds=float(offset))
                        records_in.append(
                            {
                                "Fecha_Hora_Transaccion": ts.strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "Estacion": f"({station.code}) {station.name}",
                            }
                        )

            # --- Generate check-outs ---
            if (
                mode in ("out", "both")
                and e.estimated_lambda_out
                and e.estimated_lambda_out > 0
            ):
                for i in range(sub_intervals):
                    start = base_time + timedelta(minutes=i * interval_minutes)
                    n_out = rng.poisson(e.estimated_lambda_out)
                    offsets = rng.uniform(0, interval_minutes * 60, size=n_out)
                    for offset in offsets:
                        ts = start + timedelta(seconds=float(offset))
                        records_out.append(
                            {
                                "Fecha_Hora_Transaccion": ts.strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "Estacion": f"({station.code}) {station.name}",
                            }
                        )

    date_str = target_date.strftime("%Y%m%d")

    # --- Save results ---
    if records_in:
        df_in = pd.DataFrame(records_in)
        ensure_dir("data/synthetic/check_ins")
        df_in.to_csv(f"data/synthetic/check_ins/{date_str}.csv", index=False)
        print(f"✅ Synthetic check-ins → data/synthetic/check_ins/{date_str}.csv")

    if records_out:
        df_out = pd.DataFrame(records_out)
        ensure_dir("data/synthetic/check_outs")
        df_out.to_csv(f"data/synthetic/check_outs/{date_str}.csv", index=False)
        print(f"✅ Synthetic check-outs → data/synthetic/check_outs/{date_str}.csv")

    if not records_in and not records_out:
        print("⚠️ No synthetic data generated — missing estimates?")

    session.close()


# --- Example usage ---
if __name__ == "__main__":
    target_date = datetime(2025, 10, 14)
    mode = "both"  # "in", "out", or "both"

    generate_synthetic_data_for_date(target_date, mode=mode, seed=SEED)
