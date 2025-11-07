from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.orm import Session

from src.constants.seed import SEED
from src.db.session_v1 import SessionLocal
from src.repo.v1.estimates.models import Estimate
from src.repo.v1.stations.repository import StationRepository
from src.utils.day_type import get_day_type


def simulate_poisson_series(lambdas: list[float], seed: int | None = None):
    """
    Generate simulated counts for each interval using a Poisson distribution.
    Each λ represents the mean number of arrivals/departures in that interval.
    """
    rng = np.random.default_rng(seed)
    return rng.poisson(lambdas)


def simulate_station_flows(
    station_code: str,
    date_str: str,  # e.g. "2025-10-15"
    mode: str = "both",  # "in", "out", or "both"
    seed: int | None = SEED,
):
    """
    Simulate check-in/check-out flows using time-varying Poisson rates (λ).

    Args:
        station_code: e.g., "09104"
        date_str: ISO-format date string (YYYY-MM-DD)
        mode: "in", "out", or "both"
        seed: random seed for reproducibility
    """
    session: Session = SessionLocal()
    station_repo = StationRepository(session)

    # Parse date
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year, month, day = date.year, date.month, date.day
    day_of_week = date.weekday()
    date_type = get_day_type(date)

    station = station_repo.get_station_by_code(station_code)
    if not station:
        print(f"⚠️ Station with code {station_code} not found.")
        session.close()
        return

    estimates = (
        session.query(Estimate)
        .filter(
            Estimate.station_id == station.id,
            Estimate.year == year,
            Estimate.month == month,
            Estimate.day == day,
            Estimate.day_of_week == day_of_week,
            Estimate.date_type == date_type,
        )
        .order_by(Estimate.time.asc())
        .all()
    )

    if not estimates:
        print(
            f"⚠️ No estimates found for station {station_code} ({station.name}) "
            f"on {date_str} ({date_type})."
        )
        session.close()
        return

    times = [e.time for e in estimates]
    lambdas_in = np.array([e.estimated_lambda_in or 0 for e in estimates])
    lambdas_out = np.array([e.estimated_lambda_out or 0 for e in estimates])

    plt.figure(figsize=(12, 6))

    # --- Simulate and plot λ_in ---
    if mode in ("in", "both"):
        sim_in = simulate_poisson_series(lambdas_in, seed)
        plt.plot(
            times, lambdas_in, "b-", label="Real λ_in (mean arrivals)", linewidth=1.5
        )
        plt.plot(times, sim_in, "b--", label="Simulated arrivals", alpha=0.7)

    # --- Simulate and plot λ_out ---
    if mode in ("out", "both"):
        sim_out = simulate_poisson_series(lambdas_out, seed)
        plt.plot(
            times,
            lambdas_out,
            "r-",
            label="Real λ_out (mean departures)",
            linewidth=1.5,
        )
        plt.plot(times, sim_out, "r--", label="Simulated departures", alpha=0.7)

    # --- Plot setup ---
    title = (
        f"Poisson Simulation for {station.name} ({station.code})\n"
        f"{date_str} ({date_type}) | Day of week: {day_of_week} (0=Mon)"
    )
    plt.title(title)
    plt.xlabel("Time of day (HHMM, interval start)")
    plt.ylabel("Counts per interval")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Clean up x-axis ticks (hourly spacing)
    step = max(1, len(times) // 12)
    plt.xticks(
        ticks=times[::step],
        labels=[f"{t // 100:02d}:{t % 100:02d}" for t in times[::step]],
        rotation=45,
    )

    plt.show()
    session.close()


if __name__ == "__main__":
    # Example usage
    simulate_station_flows(
        station_code="07107",  # U Nacional
        date_str="2025-10-15",
        mode="both",  # "in", "out", or "both"
        seed=SEED,
    )

# Example station codes:
# 09104 → Restrepo
# 02300 → Calle 100
# 07503 → San Mateo
# 07107 → U Nacional
