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
    params_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fit_dir: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
) -> None:
    """
    Runs diagnostics (envelope plots) for the average profile model.
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
    
    # simulations are expected in fit_dir if provided, else output_dir
    search_dir = Path(fit_dir) if fit_dir else output_dir

    persistence_dir = params_path.parent if params_path else Path("src/workflow/data")
    _, sampled_stations = load_persisted_data(persistence_dir)
    
    if not station_codes and sampled_stations:
        station_codes = [s["code"] for s in sampled_stations]

    print(f"Loading actual data for backtesting (after {cutoff_date})...")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        persistence_dir=persistence_dir
    )
    df = data[count_type]
    
    # Create a proper date column for filtering
    df["date_str"] = df.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
    test_df = df[df["date_str"] > cutoff_date].copy()
    
    if test_df.empty:
        print("No data found after cutoff date. Cannot perform backtesting.")
        return

    # Load simulated data
    sim_file = search_dir / f"avg_profile_simulations_{count_type}_{dist_type}.csv"
    if not sim_file.exists():
        raise FileNotFoundError(f"Simulated data not found: {sim_file}. Run simulation step first.")
        
    sim_df = pd.read_csv(sim_file)
    sim_df["station_code"] = sim_df["station_code"].astype(str).str.zfill(5)
    
    time_cols = [c for c in sim_df.columns if c.startswith("t_")]
    def to_hours(v):
        v = int(v.replace("t_", ""))
        return v // 100 + (v % 100) / 60.0
    time_hours = np.array([to_hours(tc) for tc in time_cols])
    
    print(f"Generating envelope plots in {output_dir}...")
    
    stations = test_df["station_code"].unique()
    if not day_types or len(day_types) == 0:
        day_types = test_df["date_type"].unique()
    
    for sc in stations:
        for dt in day_types:
            plot_envelope(test_df, sim_df, sc, dt, time_cols, time_hours, output_dir, count_type, dist_type)

    print(f"Diagnostics completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostics for Average Profile Baseline Model")
    parser.add_argument("--count_type", choices=["checkins", "checkouts"], default="checkins")
    parser.add_argument("--dist_type", choices=["poisson", "neg_binomial"], default="poisson")
    parser.add_argument("--params", type=str, help="Path to params.json")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--fit_dir", type=str, help="Directory containing simulated data")
    parser.add_argument("--stations", nargs="+", help="Subset of station codes")
    parser.add_argument("--day_types", nargs="+", help="Subset of day types")
    
    args = parser.parse_args()
    run_avg_profile_diagnostics(
        count_type=args.count_type, 
        dist_type=args.dist_type,
        params_path=Path(args.params) if args.params else None,
        output_dir=args.output_dir,
        fit_dir=args.fit_dir,
        station_codes=args.stations,
        day_types=args.day_types
    )
