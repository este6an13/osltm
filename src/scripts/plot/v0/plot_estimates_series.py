import matplotlib.pyplot as plt
from sqlalchemy.orm import Session

from src.db.session_v0 import SessionLocal
from src.repo.v0.estimates.models import Estimate
from src.repo.v0.stations.repository import StationRepository


def plot_estimate_series(
    station_code: str,
    year: int,
    month: int,
    week_of_month: int,
    day_of_week: int,
    mode: str = "both",  # "in", "out", or "both"
):
    """
    Plot estimated arrivals (λ_in) and/or departures (λ_out) per interval for a given station.

    Args:
        station_code: e.g., "09104"
        year: e.g., 2025
        month: e.g., 10 (October)
        week_of_month: e.g., 3
        day_of_week: e.g., 1 (0=Mon, 1=Tue, ..., 6=Sun)
        mode: "in", "out", or "both"
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

    # Prepare data for plotting
    times = [e.time for e in estimates]
    lambdas_in = [e.estimated_lambda_in for e in estimates]
    lambdas_out = [e.estimated_lambda_out for e in estimates]

    plt.figure(figsize=(12, 6))

    if mode in ("in", "both"):
        plt.plot(
            times,
            lambdas_in,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            label="λ_in (arrivals)",
        )

    if mode in ("out", "both"):
        plt.plot(
            times,
            lambdas_out,
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label="λ_out (departures)",
        )

    title_parts = []
    if mode == "in":
        title_parts.append("Average arrivals per interval (λ_in)")
    elif mode == "out":
        title_parts.append("Average departures per interval (λ_out)")
    else:
        title_parts.append("Arrivals and Departures per interval (λ_in / λ_out)")

    plt.title(
        f"{' | '.join(title_parts)}\n"
        f"{station.name} ({station.code}) — {year}-{month:02d} | Week {week_of_month}, Day {day_of_week} (0=Mon)"
    )
    plt.xlabel("Time of day (HHMM, interval start)")
    plt.ylabel("Average flow (λ per interval)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Make x-axis readable
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
    # Modes: "in", "out", or "both"
    plot_estimate_series(
        station_code="09104",
        year=2025,
        month=10,
        week_of_month=3,
        day_of_week=1,
        mode="both",
    )
