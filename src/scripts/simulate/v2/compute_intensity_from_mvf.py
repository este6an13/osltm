from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from sqlalchemy.orm import Session

from src.db.session_v2 import SessionLocal
from src.repo.v2.counts_15min.models import Counts15Min
from src.repo.v2.stations.repository import StationRepository


def calculate_mean_value_and_intensity(
    station_code: str,
    date_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    smoothing_factor: float = 5.0,
    plot: bool = True,
):
    """
    For a given station, compute:
        1. Mean value function m(t) = E[N(t)] across days
        2. Intensity λ(t) = dm(t)/dt for both counts_in and counts_out
    """
    session: Session = SessionLocal()
    station_repo = StationRepository(session)

    # --- Validate station ---
    station = station_repo.get_station_by_code(station_code)
    if not station:
        print(f"⚠️ Station with code {station_code} not found.")
        session.close()
        return

    # --- Build base query ---
    q = session.query(Counts15Min).filter(
        Counts15Min.station_id == station.id,
        Counts15Min.date_type == date_type,
        Counts15Min.time >= 400,
        Counts15Min.time <= 2300,
    )

    # apply date range filters if provided
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
        print(
            f"📅 Restricting to date range: {start_date or 'beginning'} → {end_date or 'end'}"
        )

    # --- Query all rows and order by date/time ---
    rows = q.order_by(
        Counts15Min.year, Counts15Min.month, Counts15Min.day, Counts15Min.time
    ).all()

    if not rows:
        print(
            f"⚠️ No counts found for {station_code} ({station.name}) and date_type '{date_type}'."
        )
        session.close()
        return

    # --- Group data per day and time window ---
    from collections import defaultdict

    daily = defaultdict(lambda: defaultdict(lambda: {"count_in": 0, "count_out": 0}))
    for r in rows:
        key = (r.year, r.month, r.day)
        daily[key][r.time]["count_in"] = int(r.count_in or 0)
        daily[key][r.time]["count_out"] = int(r.count_out or 0)

    # --- Gather all unique times and sort ---
    times = sorted({t for d in daily.values() for t in d.keys()})

    # --- Convert each day’s data into arrays ---
    in_by_day = []
    out_by_day = []
    for day_vals in daily.values():
        in_arr = np.array(
            [day_vals[t]["count_in"] if t in day_vals else 0 for t in times]
        )
        out_arr = np.array(
            [day_vals[t]["count_out"] if t in day_vals else 0 for t in times]
        )
        in_by_day.append(in_arr)
        out_by_day.append(out_arr)

    in_by_day = np.stack(in_by_day)
    out_by_day = np.stack(out_by_day)

    # --- Compute cumulative counts per day ---
    cum_in_by_day = np.cumsum(in_by_day, axis=1)
    cum_out_by_day = np.cumsum(out_by_day, axis=1)

    # --- Compute mean cumulative counts (mean value function) across days ---
    m_in = np.mean(cum_in_by_day, axis=0)
    m_out = np.mean(cum_out_by_day, axis=0)

    # --- Convert time like 405, 410 → minutes since midnight ---
    t = np.array([(t_val // 100) * 60 + (t_val % 100) for t_val in times])

    spline_in = UnivariateSpline(t, m_in, s=smoothing_factor)
    spline_out = UnivariateSpline(t, m_out, s=smoothing_factor)

    # --- Evaluate smooth function and derivative (intensity) ---
    t_smooth = np.linspace(t.min(), t.max(), len(times) * 3)
    m_in_smooth = spline_in(t_smooth)
    m_out_smooth = spline_out(t_smooth)

    lambda_in = spline_in.derivative()(t_smooth)
    lambda_out = spline_out.derivative()(t_smooth)

    if plot:
        # --- Plot results ---
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, m_in, "o", label="MVF IN (data)")
        plt.plot(t_smooth, m_in_smooth, "-", label="Smoothed m_in(t)")
        plt.plot(t, m_out, "o", label="MVF OUT (data)")
        plt.plot(t_smooth, m_out_smooth, "-", label="Smoothed m_out(t)")
        plt.ylabel("Mean Value Function")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t_smooth, lambda_in, "--", label="λ_in(t) intensity")
        plt.plot(t_smooth, lambda_out, "--", label="λ_out(t) intensity")
        plt.xlabel("Time (minutes since midnight)")
        plt.ylabel("Rate (counts per minute)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "time_smooth": t_smooth,
        "m_in": m_in_smooth,
        "m_out": m_out_smooth,
        "lambda_in": lambda_in,
        "lambda_out": lambda_out,
    }


if __name__ == "__main__":
    result = calculate_mean_value_and_intensity(
        station_code="07107",
        date_type="WD",
        end_date="2024-09-30",
        smoothing_factor=10.0,
    )
