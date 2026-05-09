import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from src.workflow.data_loader import load_data, load_persisted_data
from src.workflow.data_reader import load_csv_file
from src.realtime.loaders import load_cluster_params
from src.workflow.scripts.models.cluster.core import simulate_cluster_process, extract_cluster_parameters

def _plot_poisson_fit(sizes, ax):
    if len(sizes) == 0: return
    mean_size = np.mean(sizes)
    var_size = np.var(sizes)
    max_size = int(np.max(sizes))
    bins = np.arange(0, max_size + 2) - 0.5
    ax.hist(sizes, bins=bins, density=True, alpha=0.6, color='blue', label='Empirical')
    x = np.arange(0, max_size + 1)
    pmf = stats.poisson.pmf(x, mean_size)
    ax.plot(x, pmf, 'ro-', label=fr'Poisson($\lambda={mean_size:.2f}$)')
    ax.set_title(f'Cluster Size Distribution\n(Mean={mean_size:.2f}, Var={var_size:.2f})')
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Probability')
    ax.legend()

def _plot_normal_fit(dispersions, ax):
    if len(dispersions) == 0: return
    mean_disp = np.mean(dispersions)
    std_disp = np.std(dispersions)
    ax.hist(dispersions, bins=50, density=True, alpha=0.6, color='green', label='Empirical')
    x = np.linspace(min(dispersions), max(dispersions), 100)
    pdf = stats.norm.pdf(x, loc=mean_disp, scale=std_disp)
    ax.plot(x, pdf, 'r-', lw=2, label=fr'Normal($\mu={mean_disp:.2f}, \sigma={std_disp:.2f}$)')
    ax.set_title('Dispersion Distribution (Offsets from Centroid)')
    ax.set_xlabel('Offset (seconds)')
    ax.set_ylabel('Density')
    ax.legend()
    
def _plot_qq_normal(dispersions, ax):
    if len(dispersions) == 0: return
    stats.probplot(dispersions, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot vs Normal')

def _test_nhpp_centroids(centroids, T_total, dt_sec, num_days, ax):
    if len(centroids) < 2: return
    num_blocks = int(T_total / dt_sec)
    bin_indices = (centroids // dt_sec).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_blocks - 1)
    counts = np.bincount(bin_indices, minlength=num_blocks)
    rate_per_sec = (counts / num_days) / dt_sec
    x_hours = np.arange(num_blocks) * (dt_sec / 3600) + 4.0
    ax.bar(x_hours, rate_per_sec * 3600, width=dt_sec/3600, align='edge', alpha=0.7, color='purple')
    ax.set_title('Centroid Intensity Profile (NHPP)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Centroids per Hour')

def generate_distribution_diagnostics(
    station_code: str, day_type: str, count_type: str, 
    time_min: int, time_max: int, dt_sec: int, T_total: int,
    eps: float, min_samples: int, valid_dates: list, output_dir: Path
):
    start_sec = (time_min // 100) * 3600 + (time_min % 100) * 60
    end_sec = (time_max // 100) * 3600 + (time_max % 100) * 60
    raw_dir = Path("data/check_ins/daily") if count_type == "checkins" else Path("data/check_outs/daily")
    
    all_centroids, all_sizes, all_dispersions = [], [], []
    rng = np.random.default_rng(42)
    
    for date_str in valid_dates:
        csv_path = raw_dir / f"{date_str}.csv"
        if not csv_path.exists(): continue
        try:
            df_raw = load_csv_file(csv_path, station_codes=[station_code], count_type=count_type, include_time_components=True)
            df_raw["sec"] = df_raw["hour"] * 3600 + df_raw["minute"] * 60 + df_raw["second"]
            st_data = df_raw[df_raw["station_code"] == station_code].copy()
            if len(st_data) < 5: continue
            
            t_mask = (st_data["sec"] >= start_sec) & (st_data["sec"] <= end_sec)
            t_shifted = st_data.loc[t_mask, "sec"].sort_values().values - start_sec
            
            if count_type == "checkouts":
                t_shifted = np.sort(np.clip(t_shifted + rng.uniform(0.0, dt_sec, size=len(t_shifted)), 0.0, T_total))
                
            res = extract_cluster_parameters(t_shifted, eps=eps, min_samples=min_samples)
            all_centroids.extend(res['centroids'])
            all_sizes.extend(res['sizes'])
            all_dispersions.extend(res['dispersions'])
        except Exception:
            continue

    if not all_sizes: return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Distribution Diagnostics: {station_code} ({day_type}) - {count_type}', fontsize=16)
    _plot_poisson_fit(np.array(all_sizes), axes[0, 0])
    _plot_normal_fit(np.array(all_dispersions), axes[0, 1])
    _plot_qq_normal(np.array(all_dispersions), axes[1, 1])
    _test_nhpp_centroids(np.array(all_centroids), T_total, dt_sec, len(valid_dates), axes[1, 0])
    plt.tight_layout()
    plt.savefig(output_dir / f"distributions_{station_code}_{day_type}_{count_type}.png", dpi=200, bbox_inches="tight")
    plt.close()

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
    plt.fill_between(time_hours, sim_p05, sim_p95, color="blue", alpha=0.1, label="90% Envelope (Sim)")
    plt.fill_between(time_hours, sim_p25, sim_p75, color="blue", alpha=0.2, label="50% Envelope (Sim)")
    plt.plot(time_hours, sim_mean, "b-", linewidth=2, label="Simulated Mean")
    
    actual_counts = s_test[time_cols].values
    for i, day in enumerate(actual_counts):
        label = "Actual Test Days" if i == 0 else None
        plt.plot(time_hours, day, "k-", alpha=0.3, linewidth=0.8, label=label)
    
    actual_mean = np.mean(actual_counts, axis=0)
    plt.plot(time_hours, actual_mean, "r--", linewidth=1.5, label="Actual Test Mean")

    plt.title(f"Cluster Process Envelope — {station_code} ({day_type})\nCounts: {count_type.capitalize()}", fontweight="bold")
    plt.xlabel("Time of Day (hours)")
    plt.ylabel("Passenger Counts")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    
    fname = output_dir / f"envelope_{station_code}_{day_type}_{count_type}_cluster.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()

def run_cluster_simulate(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    n_days: int = 30,
    params_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fit_dir: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    if params_path is None:
        params_path = Path("src/workflow/params.json")
        
    with open(params_path) as f:
        params = json.load(f)
        
    cluster_params = params.get("cluster", {})
    cutoff_date = cluster_params.get("cutoff_date", "2025-11-30")
    
    if output_dir is None:
        output_dir = Path(cluster_params.get("output_dir", "src/workflow/results/cluster_fit"))
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if fit_dir is None:
        fit_dir = output_dir
    else:
        fit_dir = Path(fit_dir)
        
    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)
    
    T_total = ((time_max // 100) * 3600 + (time_max % 100) * 60) - ((time_min // 100) * 3600 + (time_min % 100) * 60)
    dt_sec = time_step * 60
    num_blocks = T_total // dt_sec
    
    # Generate time columns to match actual data
    time_cols = []
    time_hours = []
    for i in range(num_blocks):
        m = (time_min % 100) + (i * time_step)
        h = (time_min // 100) + (m // 60)
        m = m % 60
        time_cols.append(f"t_{h * 100 + m}")
        time_hours.append(h + m / 60.0)
    time_hours = np.array(time_hours)

    all_simulations = []
    print(f"Simulating {n_days} days per station/day_type using Cluster Process...")
    
    # We load params for each day type provided
    day_types_to_run = day_types if day_types else ["WD", "SA", "SU", "HO"]
    
    rng = np.random.default_rng(params.get("seed", 42))
    
    for dt in day_types_to_run:
        try:
            csv_path = fit_dir / f"cluster_params_{count_type}.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found.")
                continue
                
            df_params = pd.read_csv(csv_path)
            scs = station_codes if station_codes else df_params[df_params["day_type"] == dt]["station_code"].astype(str).str.zfill(5).unique().tolist()
            if not scs:
                continue
                
            p_map = load_cluster_params(fit_dir, scs, dt, count_type)
        except Exception as e:
            print(f"Skipping {dt}: {e}")
            continue
            
        for sc, p in p_map.items():
            sim_p = {**p, "dt_sec": dt_sec, "T_total": T_total}
            for d in range(n_days):
                # simulate events (array of seconds)
                events = simulate_cluster_process(sim_p, rng)
                
                # bin into 15 min blocks
                if len(events) > 0:
                    bin_indices = (events // dt_sec).astype(int)
                    bin_indices = np.clip(bin_indices, 0, num_blocks - 1)
                    counts = np.bincount(bin_indices, minlength=num_blocks)
                else:
                    counts = np.zeros(num_blocks, dtype=int)
                    
                res = {"station_code": sc, "day_type": dt, "sim_day": d}
                for i, col in enumerate(time_cols):
                    res[col] = counts[i]
                all_simulations.append(res)
                
    if not all_simulations:
        print("No simulations generated. Check if fit data exists.")
        return pd.DataFrame()
        
    sim_df = pd.DataFrame(all_simulations)
    output_file = output_dir / f"cluster_simulations_{count_type}.csv"
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
        stations = test_df["station_code"].unique()
        dt_to_plot = day_types if day_types and len(day_types) > 0 else test_df["date_type"].unique()
        
        for sc in stations:
            for dt in dt_to_plot:
                plot_envelope(test_df, sim_df, sc, dt, time_cols, time_hours, output_dir, count_type)
                
                # Run distribution diagnostics for the first station-day combination to avoid long processing
                valid_dates = test_df[test_df["date_type"] == dt]["date_str"].str.replace("-", "").unique().tolist()
                generate_distribution_diagnostics(
                    sc, dt, count_type, time_min, time_max, dt_sec, T_total,
                    cluster_params.get("dbscan_eps", 60), cluster_params.get("dbscan_min_samples", 3),
                    valid_dates, output_dir
                )
                
        print(f"Diagnostics completed. Plots saved to {output_dir}")

    return sim_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate & Diagnostics for Cluster Process Model")
    parser.add_argument("--count_type", choices=["checkins", "checkouts"], default="checkins")
    parser.add_argument("--n_days", type=int, default=30)
    parser.add_argument("--params", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--fit_dir", type=str)
    parser.add_argument("--stations", nargs="+")
    parser.add_argument("--day_types", nargs="+")
    
    args = parser.parse_args()
    run_cluster_simulate(
        count_type=args.count_type, 
        n_days=args.n_days,
        params_path=Path(args.params) if args.params else None,
        output_dir=args.output_dir,
        fit_dir=args.fit_dir,
        station_codes=args.stations,
        day_types=args.day_types
    )
