"""
Average Profile Model - Step 2: Simulation & Diagnostics
Generates synthetic days and produces diagnostic envelope plots.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from src.workflow.data_loader import load_data, load_persisted_data

def simulate_day(means: np.ndarray, stds: np.ndarray, dist_type: str = "poisson") -> np.ndarray:
    """Simulate one day of counts bin-wise."""
    if dist_type == "poisson":
        return np.random.poisson(means)
    elif dist_type == "neg_binomial":
        variances = stds ** 2
        simulated = np.zeros_like(means)
        for i in range(len(means)):
            m = means[i]
            v = variances[i]
            if m <= 0:
                simulated[i] = 0
            elif v <= m:
                simulated[i] = np.random.poisson(m)
            else:
                p = m / v
                n = (m ** 2) / (v - m)
                simulated[i] = stats.nbinom.rvs(n, p)
        return simulated
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

def plot_envelope(
    test_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    station_code: str,
    day_type: str,
    time_cols: list[str],
    time_hours: np.ndarray,
    output_dir: Path,
    count_type: str,
    dist_type: str
):
    """Plot simulation envelopes vs actual test data."""
    s_test = test_df[(test_df["station_code"] == station_code) & (test_df["date_type"] == day_type)]
    s_sim = sim_df[(sim_df["station_code"] == station_code) & (sim_df["day_type"] == day_type)]
    
    if s_test.empty or s_sim.empty:
        return

    sim_counts = s_sim[time_cols].values
    sim_mean = np.mean(sim_counts, axis=0)
    sim_p05 = np.percentile(sim_counts, 5, axis=0)
    sim_p95 = np.percentile(sim_counts, 95, axis=0)
    sim_p25 = np.percentile(sim_counts, 25, axis=0)
    sim_p75 = np.percentile(sim_counts, 75, axis=0)

    plt.figure(figsize=(12, 6))
    plt.fill_between(time_hours, sim_p05, sim_p95, color="blue", alpha=0.1, label="90% Envelope (Sim)")
    plt.fill_between(time_hours, sim_p25, sim_p75, color="blue", alpha=0.2, label="50% Envelope (Sim)")
    plt.plot(time_hours, sim_mean, "b-", linewidth=2, label="Simulated Mean")
    
    actual_counts = s_test[time_cols].values
    for i, day in enumerate(actual_counts):
        label = "Actual Test Days" if i == 0 else None
        plt.plot(time_hours, day, "k-", alpha=0.3, linewidth=0.8, label=label)
    
    actual_mean = np.mean(actual_counts, axis=0)
    plt.plot(time_hours, actual_mean, "r--", linewidth=1.5, label="Actual Test Mean")

    plt.title(f"Baseline (Avg Profile) Envelope — {station_code} ({day_type})\nDist: {dist_type}, Counts: {count_type.capitalize()}", fontweight="bold")
    plt.xlabel("Time of Day (hours)")
    plt.ylabel("Passenger Counts")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    
    fname = output_dir / f"envelope_{station_code}_{day_type}_{count_type}_{dist_type}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

def run_avg_profile_simulate(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    dist_type: Literal["poisson", "neg_binomial"] = "poisson",
    n_days: int = 30,
    params_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fit_dir: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Simulates days and generates diagnostics plots.
    """
    if params_path is None:
        params_path = Path("src/workflow/params.json")
        
    with open(params_path) as f:
        params = json.load(f)
        
    avg_params = params.get("avg_profile", {})
    cutoff_date = avg_params.get("cutoff_date", "2025-11-30")
    
    if output_dir is None:
        output_dir = Path(avg_params.get("output_dir", "src/workflow/results/avg_profile"))
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if fit_dir is None:
        fit_dir = output_dir
    else:
        fit_dir = Path(fit_dir)

    param_file = fit_dir / f"avg_profile_params_{count_type}.csv"
    if not param_file.exists():
        raise FileNotFoundError(f"Average profile parameters not found: {param_file}. Run fit step first.")
        
    df_params = pd.read_csv(param_file)
    df_params["station_code"] = df_params["station_code"].astype(str).str.zfill(5)
    
    if station_codes and len(station_codes) > 0:
        df_params = df_params[df_params["station_code"].isin(station_codes)].copy()
    if day_types and len(day_types) > 0:
        df_params = df_params[df_params["day_type"].isin(day_types)].copy()
        
    time_cols = [c.replace("_mean", "") for c in df_params.columns if c.endswith("_mean")]
    
    all_simulations = []
    print(f"Simulating {n_days} days per station/day_type using {dist_type}...")
    
    for _, row in df_params.iterrows():
        station_code = row["station_code"]
        day_type = row["day_type"]
        means = row[[f"{c}_mean" for c in time_cols]].values.astype(float)
        stds = row[[f"{c}_std" for c in time_cols]].values.astype(float)
        
        for d in range(n_days):
            counts = simulate_day(means, stds, dist_type=dist_type)
            res = {"station_code": station_code, "day_type": day_type, "sim_day": d}
            for i, col in enumerate(time_cols):
                res[col] = counts[i]
            all_simulations.append(res)
            
    sim_df = pd.DataFrame(all_simulations)
    output_file = output_dir / f"avg_profile_simulations_{count_type}_{dist_type}.csv"
    sim_df.to_csv(output_file, index=False)
    print(f"Saved {len(sim_df)} simulated days to {output_file}")

    # --- Diagnostics ---
    print(f"Running diagnostics (backtesting after {cutoff_date})...")
    persistence_dir = params_path.parent if params_path else Path("src/workflow/data")
    actual_data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        persistence_dir=persistence_dir
    )
    df_actual = actual_data[count_type]
    df_actual["date_str"] = df_actual.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
    test_df = df_actual[df_actual["date_str"] > cutoff_date].copy()

    if test_df.empty:
        print("No data found after cutoff date for backtesting.")
    else:
        def to_hours(v):
            v = int(v.replace("t_", ""))
            return v // 100 + (v % 100) / 60.0
        time_hours = np.array([to_hours(tc) for tc in time_cols])
        
        stations = test_df["station_code"].unique()
        dt_to_plot = day_types if day_types and len(day_types) > 0 else test_df["date_type"].unique()
        
        for sc in stations:
            for dt in dt_to_plot:
                plot_envelope(test_df, sim_df, sc, dt, time_cols, time_hours, output_dir, count_type, dist_type)
        print(f"Diagnostics completed. Plots saved to {output_dir}")

    return sim_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate & Diagnostics for Average Profile Model")
    parser.add_argument("--count_type", choices=["checkins", "checkouts"], default="checkins")
    parser.add_argument("--dist_type", choices=["poisson", "neg_binomial"], default="poisson")
    parser.add_argument("--n_days", type=int, default=30)
    parser.add_argument("--params", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--fit_dir", type=str)
    parser.add_argument("--stations", nargs="+")
    parser.add_argument("--day_types", nargs="+")
    
    args = parser.parse_args()
    run_avg_profile_simulate(
        count_type=args.count_type, 
        dist_type=args.dist_type, 
        n_days=args.n_days,
        params_path=Path(args.params) if args.params else None,
        output_dir=args.output_dir,
        fit_dir=args.fit_dir,
        station_codes=args.stations,
        day_types=args.day_types
    )
