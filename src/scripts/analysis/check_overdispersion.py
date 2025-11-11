import os
import sys

import pandas as pd


def calculate_overdispersion_per_station(df: pd.DataFrame, station_code: str):
    """Calculate overdispersion (variance/mean) in 15-min bins for a given station."""
    if df.empty:
        print("⚠️ No data found for this station.")
        return pd.DataFrame()

    # --- Convert and filter times ---
    df["timestamp"] = pd.to_datetime(df["Fecha_Transaccion"])
    df["time_int"] = df["timestamp"].dt.hour * 100 + df["timestamp"].dt.minute

    df = df[(df["time_int"] >= 400) & (df["time_int"] <= 2300)]

    # --- Count arrivals in 5-min windows ---
    df["window_5min"] = df["timestamp"].dt.floor("5min")
    per_5min_counts = df.groupby("window_5min").size().reset_index(name="count_5min")

    # --- Aggregate to 15-min windows ---
    per_5min_counts["window_15min"] = per_5min_counts["window_5min"].dt.floor("15min")

    # Compute mean, variance, and overdispersion for each 15-min window
    stats = (
        per_5min_counts.groupby("window_15min")["count_5min"]
        .agg(["mean", "var", "count"])
        .reset_index()
    )
    stats["overdispersion"] = stats["var"] / stats["mean"]
    stats["station_code"] = station_code

    return stats


def main(date_str: str, station_code: str):
    folder_path = "data/check_ins/daily"
    file_path = os.path.join(folder_path, f"{date_str}.csv")

    if not os.path.exists(file_path):
        print(f"❌ File not found for date {date_str}: {file_path}")
        sys.exit(1)

    print(f"📂 Loading data for {date_str} — Station {station_code}")
    df = pd.read_csv(
        file_path,
        usecols=["Fecha_Transaccion", "Estacion_Parada"],
        parse_dates=["Fecha_Transaccion"],
    )

    # Filter for this specific station code
    mask = df["Estacion_Parada"].astype(str).str.contains(station_code, case=False)
    df_station = df[mask]

    results = calculate_overdispersion_per_station(df_station, station_code)

    if results.empty:
        print("⚠️ No valid data for that station and date.")
        return

    # Display results
    print("\n🧾 Overdispersion summary:")
    print(results[["window_15min", "mean", "var", "overdispersion"]].head(20))

    mean_overdispersion = results["overdispersion"].mean()
    print(
        f"\n📈 Average overdispersion ratio for {station_code}: {mean_overdispersion:.2f}"
    )

    if mean_overdispersion > 1.2:
        print("💡 Suggest using a Negative Binomial model (overdispersion detected).")
    elif mean_overdispersion < 0.8:
        print("💡 Possible underdispersion (Poisson may overestimate variance).")
    else:
        print("✅ Poisson model likely adequate (dispersion ~1).")


if __name__ == "__main__":
    main("20251104", "07107")
