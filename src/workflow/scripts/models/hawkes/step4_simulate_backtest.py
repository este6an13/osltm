"""
Hawkes Backtesting — Cutoff-aware Simulation & Diagnostics

For each (station, day_type), simulates from the fitted Hawkes process
using median parameters across pre-cutoff days, then compares against
post-cutoff (test) observed data via envelope plots.

Uses the exact branching-process simulator from core.py.

Outputs:
  - hawkes_simulations_{ct}.csv           — binned counts per simulated day
  - hawkes_envelope_{s}_{d}_{ct}.png     — diagnostics envelope plots
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.workflow.data_loader import load_data, load_persisted_data
from src.workflow.scripts.models.hawkes.core import simulate_hawkes_branching

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


def plot_envelope(
    test_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    station_code: str,
    day_type: str,
    time_cols: list[str],
    time_hours: np.ndarray,
    output_dir: Path,
    count_type: str,
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
    plt.fill_between(time_hours, sim_p05, sim_p95, color="red", alpha=0.1, label="90% Envelope (Sim)")
    plt.fill_between(time_hours, sim_p25, sim_p75, color="red", alpha=0.2, label="50% Envelope (Sim)")
    plt.plot(time_hours, sim_mean, "r-", linewidth=2, label="Simulated Mean")

    actual_counts = s_test[time_cols].values
    for i, day in enumerate(actual_counts):
        label = "Actual Test Days" if i == 0 else None
        plt.plot(time_hours, day, "k-", alpha=0.3, linewidth=0.8, label=label)

    actual_mean = np.mean(actual_counts, axis=0)
    plt.plot(time_hours, actual_mean, "b--", linewidth=1.5, label="Actual Test Mean")

    plt.xlabel("Time of Day (hours)")
    plt.ylabel("Passenger Counts")
    dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
    plt.title(f"Hawkes Simulation — {station_code} ({dt_label})\nCounts: {count_type.capitalize()}", fontweight="bold")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)

    fname = output_dir / f"hawkes_envelope_{station_code}_{day_type}_{count_type}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def run_hawkes_simulate_backtest(
    n_days: int = 30,
    count_type: str = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    cutoff_date: Optional[str] = None,
    fit_dir: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
    seed: int = 42,
) -> dict:
    """
    Run cutoff-aware Hawkes simulation and diagnostics.

    Args:
        n_days: Number of simulated days per (station, day_type)
        count_type: "checkins" or "checkouts"
        cutoff_date: YYYY-MM-DD cutoff (default from params.json)
        fit_dir: Path to Hawkes fitted params CSV
        station_codes: Optional station filter
        day_types: Optional day-type filter (WD, SA, SU, HO)
        seed: RNG seed
    """
    if params_path is None:
        params_path = Path("src/workflow/params.json")

    with open(params_path) as f:
        params = json.load(f)

    step4 = params.get("step4", {})
    time_min = step4.get("time_min", 400)
    time_max = step4.get("time_max", 2300)
    time_step = step4.get("time_step", 15)
    dt_sec = time_step * 60.0

    hawkes_params = params.get("hawkes", {})
    if cutoff_date is None:
        cutoff_date = hawkes_params.get("cutoff_date", "2025-11-30")
    if output_dir is None:
        output_dir = Path(hawkes_params.get("output_dir", "src/workflow/results/hawkes_backtest"))
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    persistence_dir = params_path.parent
    _, sampled_stations = load_persisted_data(persistence_dir)
    if sampled_stations:
        available = [s["code"] for s in sampled_stations]
        station_codes = (
            [sc for sc in station_codes if sc in available]
            if station_codes
            else available
        )
    elif station_codes is None:
        raise ValueError("No station codes available.")

    if fit_dir is None:
        fit_dir = Path("src/workflow/results/hawkes_fit")
    fit_df = pd.read_csv(fit_dir / f"hawkes_params_{count_type}.csv")
    fit_df["station_code"] = fit_df["station_code"].astype(str).str.zfill(5)
    fit_df["date"] = fit_df["date"].astype(str)

    cutoff_str = cutoff_date.replace("-", "")
    fit_train = fit_df[fit_df["date"] <= cutoff_str].copy()
    print(f"   Training fits after cutoff {cutoff_date}: {len(fit_train)} (from {len(fit_df)})")

    if fit_train.empty:
        raise ValueError("No fits available before cutoff date.")

    median_params = fit_train.groupby(["station_code", "day_type"]).agg({
        "kappa": "median", "alpha": "median", "beta": "median"
    }).reset_index()

    if day_types and len(day_types) > 0:
        median_params = median_params[median_params["day_type"].isin(day_types)]
    if station_codes and len(station_codes) > 0:
        median_params = median_params[median_params["station_code"].isin(station_codes)]

    print(f"   Median params for {len(median_params)} station/day_type pairs")

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

    df["date_str"] = df.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
    train_df = df[df["date_str"] <= cutoff_date].copy()
    test_df = df[df["date_str"] > cutoff_date].copy()
    print(f"   Training records: {len(train_df)}, Test records: {len(test_df)} (cutoff={cutoff_date})")

    station_profiles = compute_mu_base_profile(train_df, time_cols)

    rng = np.random.default_rng(seed)

    all_binned = []

    for _, row in median_params.iterrows():
        sc = row["station_code"]
        dt = row["day_type"]

        if sc not in station_profiles or dt not in station_profiles[sc]:
            continue

        station_name = train_df[train_df["station_code"] == sc]["station_name"].iloc[0]

        base_probs = station_profiles[sc][dt]
        mu_blocks = base_probs / dt_sec

        profile_dict = {
            "mu_blocks": mu_blocks,
            "dt_sec": dt_sec,
            "T_total": T_total,
        }

        params_array = [row["kappa"], row["alpha"], row["beta"]]

        dt_label = DATE_TYPE_LABELS.get(dt, dt)
        print(f"\n{sc}/{dt_label}: simulating {n_days} days "
              f"(kappa={row['kappa']:.1f}, alpha={row['alpha']:.4f}, beta={row['beta']:.4f})")

        for j in range(n_days):
            arrival_seconds = simulate_hawkes_branching(params_array, profile_dict, rng)
            counts, _ = np.histogram(arrival_seconds, bins=K, range=(0, T_total))

            b_row = {
                "station_code": sc,
                "station_name": station_name,
                "day_type": dt,
                "sim_day": j,
            }
            for k, col in enumerate(time_cols):
                b_row[col] = int(counts[k])
            all_binned.append(b_row)

    binned_df = pd.DataFrame(all_binned)
    binned_csv = output_dir / f"hawkes_simulations_{count_type}.csv"
    binned_df.to_csv(binned_csv, index=False)
    print(f"\nSaved {len(binned_df)} simulated day-profiles: {binned_csv}")

    if test_df.empty:
        print("No data found after cutoff date for backtesting. Skipping diagnostics.")
    else:
        print("Generating envelope plots...")
        stations_to_plot = test_df["station_code"].unique()
        dt_to_plot = day_types if day_types and len(day_types) > 0 else test_df["date_type"].unique()
        for sc in stations_to_plot:
            for dt in dt_to_plot:
                plot_envelope(test_df, binned_df, sc, dt, time_cols, time_hours, output_dir, count_type)
        print(f"Diagnostics completed. Plots saved to {output_dir}")

    return {"sim_df": binned_df}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cutoff-aware Hawkes simulation and backtesting diagnostics"
    )
    parser.add_argument("--n_days", type=int, default=30, help="Simulated days per station/day_type")
    parser.add_argument("--count_type", choices=["checkins", "checkouts"], default="checkins")
    parser.add_argument("--params", type=str, default="src/workflow/params.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cutoff_date", type=str, default=None, help="YYYY-MM-DD cutoff")
    parser.add_argument("--fit_dir", type=str, default="src/workflow/results/hawkes_fit")
    parser.add_argument("--stations", nargs="+", default=None)
    parser.add_argument("--day_types", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_hawkes_simulate_backtest(
        n_days=args.n_days,
        count_type=args.count_type,
        params_path=Path(args.params) if args.params else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        cutoff_date=args.cutoff_date,
        fit_dir=Path(args.fit_dir) if args.fit_dir else None,
        station_codes=args.stations,
        day_types=args.day_types,
        seed=args.seed,
    )
