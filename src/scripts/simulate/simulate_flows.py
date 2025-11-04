# This script could be replaced by the synthetic data generator
# This script only generates counts, not specific records

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.orm import Session

from src.db.session_v0 import SessionLocal
from src.repo.v0.estimates.models import Estimate
from src.repo.v0.stations.repository import StationRepository


def simulate_poisson_series(lambdas: list[float], seed: int | None = None):
    """
    Generate simulated counts for each interval using a Poisson distribution.
    Each λ represents the mean number of arrivals/departures in that interval.
    """
    rng = np.random.default_rng(seed)
    return rng.poisson(lambdas)


def simulate_station_flows(
    station_code: str,
    year: int,
    month: int,
    week_of_month: int,
    day_of_week: int,
    mode: str = "both",  # "in", "out", or "both"
    seed: int | None = 42,
):
    """
    Simulate check-in/check-out flows using time-varying Poisson rates (λ).

    Args:
        station_code: e.g., "09104"
        year: Year (e.g., 2025)
        month: Month (1–12)
        week_of_month: Week number (1–5)
        day_of_week: Day of week (0=Mon … 6=Sun)
        mode: "in", "out", or "both"
        seed: random seed for reproducibility
    """
    session: Session = SessionLocal()
    station_repo = StationRepository(session)

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
            Estimate.week_of_month == week_of_month,
            Estimate.day_of_week == day_of_week,
        )
        .order_by(Estimate.time.asc())
        .all()
    )

    if not estimates:
        print(
            f"⚠️ No estimates found for station {station_code} ({station.name}) "
            f"on week {week_of_month}, day {day_of_week}."
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
    title = f"Poisson Simulation for {station.name} ({station.code})\n"
    title += f"{year}-{month:02d} | Week {week_of_month} | Day {day_of_week} (0=Mon)"
    plt.title(title)
    plt.xlabel("Time of day (HHMM, interval start)")
    plt.ylabel("Counts per interval")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

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
        station_code="07107",  # clusterize
        year=2025,  # should be similar
        month=10,  # detect differences: could drop
        week_of_month=3,  # drop
        day_of_week=1,  # go weekday, saturday, sunday/holiday and detect differences
        mode="both",  # "in", "out", or "both"
        seed=123,
    )

# 09104: Restrepo
# 02300: Calle 100
# 07503: San Mateo
# 07107: U Nacional
