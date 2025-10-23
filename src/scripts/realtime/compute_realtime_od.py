from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd


def extract_station_code(station_field: str) -> str | None:
    """Extract the code from strings like '(09104) Restrepo'."""
    if not isinstance(station_field, str) or "(" not in station_field:
        return None
    return station_field.split(")")[0].replace("(", "").strip()


def compute_od_probabilities(od_matrix: dict) -> dict:
    """Compute real-time probabilities P(S2 | S1) based on current od_matrix counts."""
    probabilities = {}

    grouped = defaultdict(list)
    for (s1, s2), timestamps in od_matrix.items():
        grouped[s1].append((s2, len(timestamps)))

    for s1, destinations in grouped.items():
        total = sum(count for _, count in destinations)
        if total > 0:
            probabilities[s1] = {s2: count / total for s2, count in destinations}

    return probabilities


def simulate_realtime_od(file_path: str, window_minutes: int = 15):
    """
    Simulate real-time OD matrix updates and probability computation.

    - Reads check-in data sorted by time ascending.
    - Tracks last station and timestamp per card.
    - Updates OD matrix when a card reappears with a different station.
    - Every 15 minutes of data, cleans up old entries and recomputes probabilities.
    """
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    # Load and filter required columns
    df = pd.read_csv(file_path, parse_dates=["Fecha_Transaccion"])
    df = df[["Fecha_Transaccion", "Estacion_Parada", "Numero_Tarjeta"]].sort_values(
        "Fecha_Transaccion", ascending=True
    )

    # Extract station codes
    df["Station_Code"] = df["Estacion_Parada"].apply(extract_station_code)
    df = df.dropna(subset=["Station_Code"])  # Remove rows without valid code

    last_seen = {}  # {card_id: (timestamp, station_code)}
    od_matrix = defaultdict(list)

    window = timedelta(minutes=window_minutes)
    last_update_time = None

    n = 0
    for _, row in df.iterrows():
        card = row["Numero_Tarjeta"]
        t2 = row["Fecha_Transaccion"]
        s2 = row["Station_Code"]

        if n % 100000 == 0:
            print(f"Processing record {n:,} — {t2}")
        n += 1

        if card in last_seen:
            t1, s1 = last_seen[card]
            if s1 != s2:
                od_matrix[(s1, s2)].append(t1)

        # Update last seen info
        last_seen[card] = (t2, s2)

        # --- Perform cleanup + recompute every 15 minutes of data ---
        if last_update_time is None:
            last_update_time = t2
        elif (t2 - last_update_time) >= timedelta(minutes=15):
            cutoff = t2 - window
            for key in list(od_matrix.keys()):
                od_matrix[key] = [ts for ts in od_matrix[key] if ts >= cutoff]
                if not od_matrix[key]:
                    del od_matrix[key]

            _ = compute_od_probabilities(od_matrix)
            last_update_time = t2

    print(f"\n✅ Finished processing {len(df):,} records")


if __name__ == "__main__":
    file_path = "data/check_ins/daily/20251014.csv"
    simulate_realtime_od(file_path)
