import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def cluster_arrival_timestamps(
    df: pd.DataFrame, station_code: str, eps_seconds: int = 60, min_samples: int = 3
):
    """
    Perform DBSCAN clustering on arrival timestamps for a station.
    """

    if df.empty:
        print("⚠️ No data found for this station.")
        return pd.DataFrame()

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["Fecha_Transaccion"])
    df = df.sort_values("timestamp")
    timestamps = df["timestamp"].values

    # Convert timestamps to numeric (seconds since first observation)
    t0 = timestamps[0]
    X = np.array(
        [(t - t0).astype("timedelta64[s]").astype(int) for t in timestamps]
    ).reshape(-1, 1)

    # Run DBSCAN clustering
    db = DBSCAN(eps=eps_seconds, min_samples=min_samples).fit(X)
    df["cluster"] = db.labels_

    # Summary
    labels = df["cluster"].unique()
    num_clusters = len([c for c in labels if c != -1])
    num_noise = (df["cluster"] == -1).sum()

    print(f"\n🔍 Clustering results for station {station_code}:")
    print(f"   • DBSCAN eps     = {eps_seconds} sec")
    print(f"   • DBSCAN min_pts = {min_samples}")
    print(f"   • Number of clusters found = {num_clusters}")
    print(f"   • Noise points             = {num_noise}")

    # Show cluster intervals
    for c in sorted([c for c in labels if c != -1]):
        sub = df[df["cluster"] == c]
        start, end = sub["timestamp"].iloc[0], sub["timestamp"].iloc[-1]
        print(f"     - Cluster {c}: {len(sub)} arrivals ({start} → {end})")

    return df


def main(
    date_str: str,
    station_code: str,
    time_start: int = 400,  # optional time window start HHMM
    time_end: int = 2300,  # optional time window end HHMM
):
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

    # Filter by station code
    mask = df["Estacion_Parada"].astype(str).str.contains(station_code, case=False)
    df_station = df[mask]

    if df_station.empty:
        print("⚠️ No valid data for that station and date.")
        return

    # -------------------
    # TIME RANGE FILTER
    # -------------------
    df_station["timestamp"] = pd.to_datetime(df_station["Fecha_Transaccion"])
    df_station["time_int"] = (
        df_station["timestamp"].dt.hour * 100 + df_station["timestamp"].dt.minute
    )

    df_station = df_station[
        (df_station["time_int"] >= time_start) & (df_station["time_int"] <= time_end)
    ]

    if df_station.empty:
        print(f"⚠️ No data in the selected time range {time_start}-{time_end}.")
        return

    # -------------------
    # CLUSTERING
    # -------------------
    clustered_df = cluster_arrival_timestamps(
        df_station,
        station_code,
        eps_seconds=60,
        min_samples=3,
    )

    if clustered_df.empty:
        return

    print("\n🧾 First 20 rows with cluster labels:")
    print(clustered_df[["timestamp", "cluster"]].head(20))

    # -----------------------------------
    # PLOT 1 — Real arrivals per minute
    # -----------------------------------
    df_station["minute"] = df_station["timestamp"].dt.floor("1min")
    per_min_counts = df_station.groupby("minute").size().reset_index(name="count")

    plt.figure(figsize=(12, 4))
    plt.plot(per_min_counts["minute"], per_min_counts["count"])
    plt.title("Arrivals per Minute")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # PLOT 2 — Cluster-level summary
    # -----------------------------------
    clusters = clustered_df[clustered_df["cluster"] != -1]

    cluster_summary = (
        clusters.groupby("cluster")
        .agg(
            count=("timestamp", "size"),
            mean_time=("timestamp", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(12, 4))
    plt.scatter(cluster_summary["mean_time"], cluster_summary["count"], s=60)
    plt.title("Cluster Summary (Mean Timestamp vs Cluster Size)")
    plt.xlabel("Mean timestamp of cluster")
    plt.ylabel("Cluster size (# arrivals)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example:
    # main("20251104", "07107", 600, 2100)
    main("20251104", "07107", 500, 600)
