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


def simulate_real_time_estimates(
    mode: str, target_date: str | datetime, target_station_code: str
):
    """
    Simulate real-time λ̂ estimation for synthetic check-ins, check-outs, or both.

    Args:
        mode: One of "in", "out", or "both".
        target_date: Target date as 'YYYY-MM-DD' or datetime.
        target_station_code: Station code for plotting, e.g. '09104'.
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date).date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()

    date_str = target_date.strftime("%Y%m%d")

    # --- File paths ---
    path_in = f"data/synthetic/check_ins/{date_str}.csv"
    path_out = f"data/synthetic/check_outs/{date_str}.csv"

    # --- Determine which modes to simulate ---
    simulate_in = mode in ("in", "both")
    simulate_out = mode in ("out", "both")

    if simulate_in and not os.path.exists(path_in):
        print(f"⚠️ Check-ins file not found: {path_in}")
        simulate_in = False
    if simulate_out and not os.path.exists(path_out):
        print(f"⚠️ Check-outs file not found: {path_out}")
        simulate_out = False

    if not (simulate_in or simulate_out):
        raise FileNotFoundError("No valid CSV files found for the given date/mode.")

    print(f"📅 Simulating real-time λ̂ estimates for {target_date} (mode={mode})")

    # --- Helper to process one dataset ---
    def process_file(file_path: str, label: str):
        df = pd.read_csv(file_path, parse_dates=["Fecha_Hora_Transaccion"])
        df["station_code"] = df["Estacion"].apply(extract_station_code)
        df = df[df["Fecha_Hora_Transaccion"].dt.date == target_date]

        if df.empty:
            print(f"⚠️ No records found for {target_date} in {label}.")
            return [], []

        df.sort_values("Fecha_Hora_Transaccion", inplace=True)
        start_time = df["Fecha_Hora_Transaccion"].min().floor("5min")
        end_time = df["Fecha_Hora_Transaccion"].max().ceil("5min")

        station_state = defaultdict(
            lambda: {"recent_counts": deque(maxlen=3), "current_estimate": 0.0}
        )
        time_series = []
        estimates = []

        current_time = start_time
        while current_time < end_time:
            next_time = current_time + timedelta(minutes=5)
            window_df = df[
                (df["Fecha_Hora_Transaccion"] >= current_time)
                & (df["Fecha_Hora_Transaccion"] < next_time)
            ]

            if not window_df.empty:
                counts = window_df.groupby("station_code").size()
                for station, count in counts.items():
                    s = station_state[station]
                    s["recent_counts"].append(count)
                    s["current_estimate"] = sum(s["recent_counts"]) / len(
                        s["recent_counts"]
                    )

            # Record λ̂ for the target station every 15 minutes
            if int(current_time.minute) % 15 == 0:
                s = station_state.get(target_station_code)
                if s:
                    time_series.append(current_time)
                    estimates.append(s["current_estimate"])

            current_time = next_time

        return time_series, estimates

    # --- Run simulation(s) ---
    time_in, lambda_in = process_file(path_in, "check-ins") if simulate_in else ([], [])
    time_out, lambda_out = (
        process_file(path_out, "check-outs") if simulate_out else ([], [])
    )

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plotted = False

    if lambda_in:
        plt.plot(time_in, lambda_in, "b-o", label="λ̂_in (check-ins)", linewidth=1.5)
        plotted = True
    if lambda_out:
        plt.plot(time_out, lambda_out, "r-o", label="λ̂_out (check-outs)", linewidth=1.5)
        plotted = True

    if not plotted:
        print(f"⚠️ No data found for station {target_station_code} on {target_date}.")
        return

    plt.title(f"Rolling λ̂ estimates for station {target_station_code} ({target_date})")
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Estimated λ̂ (per 5 min avg of last 3 windows)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Example usage ---
    simulate_real_time_estimates(
        mode="both",  # "in", "out", or "both"
        target_date="2025-10-14",
        target_station_code="09104",
    )
