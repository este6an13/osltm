"""
Average Profile Model - Step 3: Diagnostics
Generates envelope plots comparing test data against simulated historical bounds.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.workflow.data_loader import load_data, load_persisted_data

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
    
    # Filter for specific station/day_type
    s_test = test_df[(test_df["station_code"] == station_code) & (test_df["date_type"] == day_type)]
    s_sim = sim_df[(sim_df["station_code"] == station_code) & (sim_df["day_type"] == day_type)]
    
    if s_test.empty or s_sim.empty:
        return

    # Compute envelopes from simulations
    sim_counts = s_sim[time_cols].values
    sim_mean = np.mean(sim_counts, axis=0)
    sim_p05 = np.percentile(sim_counts, 5, axis=0)
    sim_p95 = np.percentile(sim_counts, 95, axis=0)
    sim_p25 = np.percentile(sim_counts, 25, axis=0)
    sim_p75 = np.percentile(sim_counts, 75, axis=0)

    plt.figure(figsize=(12, 6))
    
    # Plot envelopes
    plt.fill_between(time_hours, sim_p05, sim_p95, color="blue", alpha=0.1, label="90% Envelope (Sim)")
    plt.fill_between(time_hours, sim_p25, sim_p75, color="blue", alpha=0.2, label="50% Envelope (Sim)")
    plt.plot(time_hours, sim_mean, "b-", linewidth=2, label="Simulated Mean")
    
    # Plot actual test days
    actual_counts = s_test[time_cols].values
    for i, day in enumerate(actual_counts):
        label = "Actual Test Days" if i == 0 else None
        plt.plot(time_hours, day, "k-", alpha=0.3, linewidth=0.8, label=label)
    
    # Plot actual mean for clarity
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

def run_avg_profile_diagnostics(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    dist_type: Literal["poisson", "neg_binomial"] = "poisson",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
):
    """
    Generates diagnostics plots.
    """
    if params_path is None:
        params_path = Path("src/workflow/params.json")
    
    with open(params_path) as f:
        params = json.load(f)
    
    avg_params = params.get("avg_profile", {})
    cutoff_date = avg_params.get("cutoff_date")
    if output_dir is None:
        output_dir = Path(avg_params.get("output_dir", "src/workflow/results/avg_profile"))
    
    # Load actual data
    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)
    
    persistence_dir = params_path.parent
    _, sampled_stations = load_persisted_data(persistence_dir)
    station_codes = [s["code"] for s in sampled_stations] if sampled_stations else None

    print(f"Loading actual data for backtesting (after {cutoff_date})...")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min, time_max=time_max, time_step=time_step,
    )
    df = data[count_type]
    df["date_str"] = df.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
    test_df = df[df["date_str"] > cutoff_date].copy()
    
    if test_df.empty:
        print("No data found after cutoff date. Cannot perform backtesting.")
        return

    # Load simulated data
    sim_file = output_dir / f"avg_profile_simulations_{count_type}_{dist_type}.csv"
    if not sim_file.exists():
        raise FileNotFoundError(f"Simulations not found: {sim_file}. Run step2_simulate.py first.")
    sim_df = pd.read_csv(sim_file)
    
    time_cols = [c for c in test_df.columns if c.startswith("t_")]
    
    # Convert time_cols to hours for plotting
    def to_hours(tc):
        v = int(tc.replace("t_", ""))
        return v // 100 + (v % 100) / 60.0
    time_hours = np.array([to_hours(tc) for tc in time_cols])
    
    print(f"Generating envelope plots in {output_dir}...")
    
    stations = test_df["station_code"].unique()
    for sc in stations:
        day_types = test_df[test_df["station_code"] == sc]["date_type"].unique()
        for dt in day_types:
            plot_envelope(test_df, sim_df, sc, dt, time_cols, time_hours, output_dir, count_type, dist_type)

    print(f"Diagnostics completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostics for Average Profile Baseline Model")
    parser.add_argument("--count_type", choices=["checkins", "checkouts"], default="checkins")
    parser.add_argument("--dist_type", choices=["poisson", "neg_binomial"], default="poisson")
    
    args = parser.parse_args()
    run_avg_profile_diagnostics(count_type=args.count_type, dist_type=args.dist_type)
