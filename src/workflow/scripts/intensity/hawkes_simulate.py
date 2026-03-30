"""
Hawkes Process Simulation — Sample synthetic arrivals from the fitted Hawkes model

Outputs:
  1) Raw event CSV (one row per arrival)
  2) Binned count CSV (one row per simulated day)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import argparse

import numpy as np
import pandas as pd

from src.workflow.data_loader import load_data, load_persisted_data
from src.workflow.scripts.intensity.hawkes_core import simulate_hawkes_branching

DATE_TYPE_LABELS = {"WD": "Weekday", "SA": "Saturday", "SU": "Sunday", "HO": "Holiday"}

def compute_mu_base_profile(df, time_cols):
    station_profiles = {}
    for sc in df["station_code"].unique():
        sdf = df[df["station_code"] == sc]
        station_profiles[sc] = {}
        for dt in sdf["date_type"].unique():
            dtdf = sdf[sdf["date_type"] == dt]
            if len(dtdf) < 5:
                continue
            mean_counts = dtdf[time_cols].fillna(0).mean().values
            total_mean = mean_counts.sum()
            if total_mean > 0:
                mu_base_k = mean_counts / total_mean
            else:
                mu_base_k = np.zeros(len(time_cols))
            station_profiles[sc][dt] = mu_base_k
    return station_profiles
    
def _plot_simulation_comparison(
    observed_df, simulated_df, time_cols, time_hours, output_dir, count_type
):
    import matplotlib.pyplot as plt

    pairs = simulated_df.groupby(["station_code", "date_type"]).size().index.tolist()
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    cols = min(4, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (sc, dt) in enumerate(pairs):
        ax = axes[idx // cols][idx % cols]

        obs = observed_df[
            (observed_df["station_code"] == sc) & (observed_df["date_type"] == dt)
        ][time_cols].fillna(0)
        sim = simulated_df[
            (simulated_df["station_code"] == sc) & (simulated_df["date_type"] == dt)
        ][time_cols].fillna(0)

        obs_mean = obs.mean().values
        sim_mean = sim.mean().values
        sim_std = sim.std().values

        ax.plot(time_hours, obs_mean, "k-", lw=2, label="Observed mean")
        ax.plot(time_hours, sim_mean, "r-", lw=1.5, alpha=0.8, label="Simulated mean")
        ax.fill_between(
            time_hours,
            np.maximum(0, sim_mean - 2 * sim_std), sim_mean + 2 * sim_std,
            alpha=0.15, color="red", label="Sim ±2σ",
        )

        dt_label = DATE_TYPE_LABELS.get(dt, dt)
        ax.set_title(f"{sc} {dt_label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Counts / bin")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_pairs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"Hawkes Simulation vs Observed ({count_type.capitalize()})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fname = output_dir / f"hawkes_simulation_comparison_{count_type}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Saved comparison plot: {fname}")
    plt.close()

def run_hawkes_simulate(
    n_days: int = 10,
    count_type: str = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    fit_dir: Optional[Path] = None,
    base_date: str = "20260101",
    seed: int = 42,
):
    if params_path is None: params_path = Path("src/workflow/params.json")
    with open(params_path) as f: params = json.load(f)
        
    step4 = params.get("step4", {})
    time_min = step4.get("time_min", 400)
    time_max = step4.get("time_max", 2300)
    time_step = step4.get("time_step", 15)
    
    dt_sec = time_step * 60.0
    
    persistence_dir = Path(step4.get("persistence_dir", "src/workflow/data"))
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)
    available = [s["code"] for s in sampled_stations] if sampled_stations else []
    station_codes = [sc for sc in station_codes if sc in available] if station_codes else available
    
    if fit_dir is None: fit_dir = Path("src/workflow/results/hawkes_fit")
    fit_df = pd.read_csv(fit_dir / f"hawkes_params_{count_type}.csv")
    fit_df["station_code"] = fit_df["station_code"].astype(str).str.zfill(5)
    
    # We take the median parameters over all days to simulate a "typical" profile
    median_params = fit_df.groupby(["station_code", "day_type"]).agg({
        "kappa": "median", "alpha": "median", "beta": "median"
    }).reset_index()
    
    print(f"📊 Loading raw {count_type} count data to build background profiles...")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min, time_max=time_max, time_step=time_step,
    )
    df = data[count_type]
    time_cols = sorted([c for c in df.columns if c.startswith("t_")], key=lambda c: int(c.split("_")[1]))
    K = len(time_cols)
    T_total = K * dt_sec
    time_hours = np.array([int(c.split("_")[1]) // 100 + (int(c.split("_")[1]) % 100) / 60.0 for c in time_cols])
    
    station_profiles = compute_mu_base_profile(df, time_cols)
    
    if output_dir is None:
        output_dir = Path("src/workflow/results/hawkes_simulate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_dt = datetime.strptime(base_date, "%Y%m%d")
    rng = np.random.default_rng(seed)
    
    all_events = []
    all_binned = []
    
    for _, row in median_params.iterrows():
        sc = row["station_code"]
        dt = row["day_type"]
        
        if sc not in station_codes: continue
        if sc not in station_profiles or dt not in station_profiles[sc]: continue
        
        station_name = df[df["station_code"] == sc]["station_name"].iloc[0]
        estacion_str = f"({sc}) {station_name}"
        
        base_probs = station_profiles[sc][dt]
        # intensity per second
        mu_blocks = base_probs / dt_sec
        
        profile_dict = {
            "mu_blocks": mu_blocks,
            "dt_sec": dt_sec,
            "T_total": T_total
        }
        
        params_array = [row["kappa"], row["alpha"], row["beta"]]
        
        dt_label = DATE_TYPE_LABELS.get(dt, dt)
        print(f"\n{sc}/{dt_label}: simulating {n_days} days "
              f"(kappa={row['kappa']:.1f}, alpha={row['alpha']:.4f}, beta={row['beta']:.4f})")
               
        for j in range(n_days):
            sim_date = base_dt + timedelta(days=j)
            
            # Exact Continuous Time Simulation (no rejection)
            arrival_seconds = simulate_hawkes_branching(params_array, profile_dict, rng)
            
            total = len(arrival_seconds)
            print(f"  Day {j+1}: {total} arrivals")
            
            # Subdivide into bins
            counts, _ = np.histogram(arrival_seconds, bins=K, range=(0, T_total))
            
            # Convert seconds-from-window-start to absolute datetime
            start_sec_midnight = int(time_hours[0] * 3600)
            for t_sec in arrival_seconds:
                abs_sec = int(t_sec) + start_sec_midnight
                h = abs_sec // 3600
                m = (abs_sec % 3600) // 60
                s = abs_sec % 60
                event_dt = sim_date.replace(hour=h, minute=m, second=s)
                
                all_events.append({
                    "Fecha_Transaccion": event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "Estacion_Parada": estacion_str,
                })
                
            b_row = {
                "year": sim_date.year,
                "month": sim_date.month,
                "day": sim_date.day,
                "date_type": dt,
                "station_code": sc,
                "station_name": station_name,
            }
            for k, col in enumerate(time_cols):
                b_row[col] = int(counts[k])
            all_binned.append(b_row)

    events_df = pd.DataFrame(all_events)
    events_csv = output_dir / f"hawkes_simulated_events_{count_type}.csv"
    events_df.to_csv(events_csv, index=False)
    print(f"\nSaved {len(events_df)} simulated events: {events_csv}")

    binned_df = pd.DataFrame(all_binned)
    binned_csv = output_dir / f"hawkes_simulated_binned_{count_type}.csv"
    binned_df.to_csv(binned_csv, index=False)
    print(f"Saved {len(binned_df)} simulated day-profiles: {binned_csv}")
    
    _plot_simulation_comparison(df, binned_df, time_cols, time_hours, output_dir, count_type)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_days", type=int, default=10)
    parser.add_argument("--count_type", default="checkins", choices=["checkins", "checkouts"])
    parser.add_argument("--output_dir", default="src/workflow/results/hawkes_simulate")
    parser.add_argument("--params", default="src/workflow/params.json")
    parser.add_argument("--stations", nargs="+", default=None)
    parser.add_argument("--fit_dir", default="src/workflow/results/hawkes_fit")
    parser.add_argument("--base_date", default="20260101")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_hawkes_simulate(
        n_days=args.n_days,
        count_type=args.count_type,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
        fit_dir=Path(args.fit_dir),
        base_date=args.base_date,
        seed=args.seed,
    )
