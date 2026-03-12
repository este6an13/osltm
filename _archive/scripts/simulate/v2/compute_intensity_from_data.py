from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from sqlalchemy.orm import Session

from src.db.session_v2 import SessionLocal
from src.repo.v2.counts_15min.models import Counts15Min
from src.repo.v2.stations.repository import StationRepository


def estimate_intensity_then_mvf(
    station_code: str,
    date_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    smoothing_factor: float = 5.0,
    plot: bool = False,
):
    """
    For a given station:
        1. Compute raw intensity λ(t) = mean(count)/15 for IN and OUT.
        2. Smooth λ(t) using a spline.
        3. Integrate λ(t) to get the Mean Value Function m(t).
        4. Plot:
            a. Intensity (raw and smoothed)
            b. Cumulative mean value function m(t)
    """

    session: Session = SessionLocal()
    station_repo = StationRepository(session)

    # --- Validate station ---
    station = station_repo.get_station_by_code(station_code)
    if not station:
        print(f"⚠️ Station with code {station_code} not found.")
        session.close()
        return

    # --- Base query ---
    q = session.query(Counts15Min).filter(
        Counts15Min.station_id == station.id,
        Counts15Min.date_type == date_type,
        Counts15Min.time >= 400,
        Counts15Min.time <= 2300,
    )

    # Date filtering
    if start_date:
        sd = datetime.strptime(start_date, "%Y-%m-%d")
        q = q.filter(
            (Counts15Min.year, Counts15Min.month, Counts15Min.day)
            >= (sd.year, sd.month, sd.day)
        )
    if end_date:
        ed = datetime.strptime(end_date, "%Y-%m-%d")
        q = q.filter(
            (Counts15Min.year, Counts15Min.month, Counts15Min.day)
            <= (ed.year, ed.month, ed.day)
        )
    if start_date or end_date:
        print(f"📅 Restricting to: {start_date or 'beginning'} → {end_date or 'end'}")

    rows = q.order_by(
        Counts15Min.year, Counts15Min.month, Counts15Min.day, Counts15Min.time
    ).all()

    if not rows:
        print(f"⚠️ No counts found for station {station_code}.")
        session.close()
        return

    # --- Group by day ---
    from collections import defaultdict

    daily = defaultdict(lambda: defaultdict(lambda: {"in": 0, "out": 0}))

    for r in rows:
        key = (r.year, r.month, r.day)
        daily[key][r.time]["in"] = int(r.count_in or 0)
        daily[key][r.time]["out"] = int(r.count_out or 0)

    # Unique times
    times = sorted({t for d in daily.values() for t in d.keys()})

    # Build per-day arrays
    in_by_day = []
    out_by_day = []

    for day_vals in daily.values():
        in_arr = np.array([day_vals[t]["in"] for t in times])
        out_arr = np.array([day_vals[t]["out"] for t in times])
        in_by_day.append(in_arr)
        out_by_day.append(out_arr)

    in_by_day = np.stack(in_by_day)
    out_by_day = np.stack(out_by_day)

    # --- Mean intensity (counts per minute) ---
    # Each interval is 15 minutes
    mean_in = np.mean(in_by_day, axis=0)
    mean_out = np.mean(out_by_day, axis=0)

    lambda_in_raw = mean_in / 15.0
    lambda_out_raw = mean_out / 15.0

    # Convert 405 → minutes since midnight
    t = np.array([(tt // 100) * 60 + (tt % 100) for tt in times])

    # --- Smooth intensity ---
    spline_in = UnivariateSpline(t, lambda_in_raw, s=smoothing_factor)
    spline_out = UnivariateSpline(t, lambda_out_raw, s=smoothing_factor)

    t_smooth = np.linspace(t.min(), t.max(), len(times) * 3)
    lambda_in_smooth = spline_in(t_smooth)
    lambda_out_smooth = spline_out(t_smooth)

    # --- Compute Mean Value Function by integrating λ(t) ---
    m_in = np.cumsum(lambda_in_smooth) * np.mean(np.diff(t_smooth))
    m_out = np.cumsum(lambda_out_smooth) * np.mean(np.diff(t_smooth))

    if plot:
        # --- Plot ---
        plt.figure(figsize=(12, 8))

        # --- Intensity ---
        plt.subplot(2, 1, 1)
        plt.plot(t, lambda_in_raw, "o", label="Raw λ_in (data)")
        plt.plot(t_smooth, lambda_in_smooth, "-", label="Smoothed λ_in(t)")
        plt.plot(t, lambda_out_raw, "o", label="Raw λ_out (data)")
        plt.plot(t_smooth, lambda_out_smooth, "-", label="Smoothed λ_out(t)")
        plt.ylabel("Intensity λ(t) [per minute]")
        plt.legend()

        # --- Mean Value Function ---
        plt.subplot(2, 1, 2)
        plt.plot(t_smooth, m_in, "-", label="m_in(t) from ∫λ dt")
        plt.plot(t_smooth, m_out, "-", label="m_out(t) from ∫λ dt")
        plt.xlabel("Time (minutes since midnight)")
        plt.ylabel("Mean Value Function m(t)")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return {
        "time_smooth": t_smooth,
        "lambda_in": lambda_in_smooth,
        "lambda_out": lambda_out_smooth,
        "m_in": m_in,
        "m_out": m_out,
    }


if __name__ == "__main__":
    # Example call
    result = estimate_intensity_then_mvf(
        station_code="07107",
        date_type="WD",
        end_date="2025-09-30",
        smoothing_factor=10.0,
    )
