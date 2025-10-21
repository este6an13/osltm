import os
from collections import defaultdict, deque
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd


def extract_station_code(station_field: str) -> str | None:
    """Extract the code from strings like '(09104) Restrepo'."""
    if not isinstance(station_field, str) or "(" not in station_field:
        return None
    return station_field.split(")")[0].replace("(", "").strip()


def simulate_real_time_checkins(
    csv_path: str, target_station_code: str, target_date: str | datetime
):
    """
    Simulate real-time check-in processing from a single CSV file and
    collect rolling λ̂ estimates for a given station and date.

    Args:
        csv_path: Path to the daily CSV file.
        target_station_code: Station code to monitor, e.g. '09104'.
        target_date: Target date (str 'YYYY-MM-DD' or datetime.date).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # --- Parse and normalize target date ---
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()

    # --- Load and prepare data ---
    df = pd.read_csv(csv_path, parse_dates=["Fecha_Transaccion"])
    df["station_code"] = df["Estacion_Parada"].apply(extract_station_code)

    # ✅ Filter only the records for the given date
    df = df[df["Fecha_Transaccion"].dt.date == target_date]

    if df.empty:
        print(f"⚠️ No records found for {target_date} in this file.")
        return

    df.sort_values("Fecha_Transaccion", inplace=True)

    start_time = df["Fecha_Transaccion"].min().floor("5min")
    end_time = df["Fecha_Transaccion"].max().ceil("5min")

    # --- State per station ---
    station_state = defaultdict(
        lambda: {"recent_counts": deque(maxlen=3), "current_estimate": 0.0}
    )

    # --- Tracking for plotting ---
    time_series = []
    estimates = []

    # --- Real-time simulation ---
    current_time = start_time
    while current_time < end_time:
        next_time = current_time + timedelta(minutes=5)
        window_df = df[
            (df["Fecha_Transaccion"] >= current_time)
            & (df["Fecha_Transaccion"] < next_time)
        ]

        if not window_df.empty:
            counts = window_df.groupby("station_code").size()
            for station, count in counts.items():
                s = station_state[station]
                s["recent_counts"].append(count)
                s["current_estimate"] = sum(s["recent_counts"]) / len(
                    s["recent_counts"]
                )

        # Example output: show top 3 stations by λ_in every 30 minutes
        if int(current_time.minute) % 30 == 0:
            snapshot = sorted(
                [
                    (st, round(s["current_estimate"], 2))
                    for st, s in station_state.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            print(f"🕒 {current_time.strftime('%H:%M')} — Top 3 stations by λ_in:")
            for st, val in snapshot[:3]:
                print(f"  {st}: λ̂ = {val}")

        # Every 15 minutes, record the λ̂ for the target station
        if int(current_time.minute) % 15 == 0:
            s = station_state.get(target_station_code)
            if s:
                time_series.append(current_time)
                estimates.append(s["current_estimate"])

        current_time = next_time

    # --- Plotting ---
    if not estimates:
        print(f"⚠️ No data found for station {target_station_code} on {target_date}.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(time_series, estimates, marker="o", linestyle="-", linewidth=1.5)
    plt.title(f"Rolling λ̂ estimate for station {target_station_code} ({target_date})")
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Estimated λ_in (check-ins per 5 min avg of last 3 windows)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_real_time_checkins(
        csv_path="data/check_ins/daily/20251014.csv",
        target_station_code="09104",
        target_date="2025-10-14",
    )
