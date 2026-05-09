import json
import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.workflow.data_loader import load_persisted_data
from src.workflow.data_reader import load_csv_file
from src.workflow.scripts.models.cluster.core import extract_cluster_parameters

def plot_sample_clustering(
    t_sec: np.ndarray, centroids: np.ndarray, noise: np.ndarray, 
    sizes: np.ndarray, sc: str, day_type: str, dt_sec: int, out_dir: Path
):
    """Plot a sample 1D timeline of the clustering results."""
    plt.figure(figsize=(14, 4))
    
    # Plot noise points
    if len(noise) > 0:
        plt.scatter(noise / 3600, np.zeros_like(noise), color='gray', alpha=0.5, s=10, label='Noise')
        
    # Plot clustered points (we don't have the exact points easily separated here, but we can plot centroids)
    # Actually, we can just plot the centroids with sizes mapped to marker size
    if len(centroids) > 0:
        # Scale sizes for marker size (e.g. size * 10)
        marker_sizes = sizes * 5
        plt.scatter(centroids / 3600, np.zeros_like(centroids) + 0.1, 
                    color='blue', alpha=0.6, s=marker_sizes, label='Centroids (size ~ cluster size)')
                    
        # Add vertical lines for centroids
        for c in centroids:
            plt.axvline(c / 3600, color='blue', alpha=0.2, linestyle='--', linewidth=0.5)

    plt.title(f"Sample Centroid Extraction - {sc} ({day_type})")
    plt.xlabel("Time of Day (hours)")
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"sample_clustering_{sc}_{day_type}.png", dpi=200, bbox_inches="tight")
    plt.close()

def run_cluster_fit(
    params_path: Path,
    station_codes_arg: Optional[list[str]] = None,
    date_percentage: float = 1.0,
    count_type: str = "checkins",
    output_dir: Optional[Path] = None,
    cutoff_date: Optional[str] = None,
    show_plots: bool = False,
    method: str = "dbscan",
    target_size: int = 5,
):
    with open(params_path) as f:
        params = json.load(f)
        
    cluster_params = params.get("cluster", {})
    eps = cluster_params.get("dbscan_eps", 60)
    min_samples = cluster_params.get("dbscan_min_samples", 3)
    
    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)
    
    start_sec = (time_min // 100) * 3600 + (time_min % 100) * 60
    end_sec = (time_max // 100) * 3600 + (time_max % 100) * 60
    T_total = end_sec - start_sec
    dt_sec = time_step * 60
    num_blocks = T_total // dt_sec
    
    persistence_dir = params_path.parent
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)
    
    if not sampled_dates or not sampled_stations:
        raise ValueError("No sampled dates/stations.")

    if cutoff_date is not None:
        cutoff_str = cutoff_date.replace("-", "")
        train_dates = [d for d in sampled_dates if d <= cutoff_str]
        print(f"   Training dates after cutoff {cutoff_date}: {len(train_dates)} (from {len(sampled_dates)})")
        sampled_dates = train_dates

    if date_percentage < 1.0:
        import random
        random.seed(params.get("seed", 42))
        n_sample = max(1, int(len(sampled_dates) * date_percentage))
        sampled_dates = random.sample(sampled_dates, n_sample)
        print(f"📉 Downsampled dates to {n_sample} ({date_percentage*100}%) for faster validation.")

    available_station_codes = [s["code"] for s in sampled_stations]
    if station_codes_arg:
        station_codes = [sc for sc in station_codes_arg if sc in available_station_codes]
        if not station_codes:
            print("⚠️ Provided stations were not in the sampled stations. Proceeding anyway.")
            station_codes = station_codes_arg
    else:
        station_codes = available_station_codes
        
    raw_dir = (
        Path("data/check_ins/daily")
        if count_type == "checkins"
        else Path("data/check_outs/daily")
    )
    
    # Map from (station_code, day_type) to collection of parameters
    collections = {}
    
    from src.workflow.data_loader import load_data
    print(f"📊 Loading aggregated {count_type} data to identify day types...")
    data_15min = load_data(
        dates=sampled_dates,
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min,
        time_max=time_max,
        time_step=time_step
    )
    df_15min = data_15min[count_type]
    date_to_daytype = {}
    for _, row in df_15min[["year", "month", "day", "date_type"]].drop_duplicates().iterrows():
        date_str = f"{row['year']:04d}{row['month']:02d}{row['day']:02d}"
        date_to_daytype[date_str] = row['date_type']

    print(f"🔍 Extracting cluster parameters via DBSCAN...")
    rng = np.random.default_rng(params.get("seed", 42))
    
    for date_str in sampled_dates:
        csv_path = raw_dir / f"{date_str}.csv"
        if not csv_path.exists():
            continue
            
        day_type = date_to_daytype.get(date_str, "WD")
        
        try:
            df_raw = load_csv_file(
                csv_path,
                station_codes=station_codes,
                count_type=count_type,
                include_time_components=True,
            )
            df_raw["sec"] = (
                df_raw["hour"] * 3600
                + df_raw["minute"] * 60
                + df_raw["second"]
            )
        except Exception:
            continue
            
        out_dir_path = Path(output_dir) if output_dir else Path("src/workflow/results/cluster_fit")
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        plotted_samples = set()
        
        for sc in station_codes:
            st_data = df_raw[df_raw["station_code"] == sc].copy()
            if len(st_data) < 5:
                continue
                
            t_mask = (st_data["sec"] >= start_sec) & (st_data["sec"] <= end_sec)
            t_sec_arr = st_data.loc[t_mask, "sec"].sort_values().values
            t_shifted = t_sec_arr - start_sec
            
            if count_type == "checkouts":
                # Add uniform jitter to spread the 15-min binned checkout data
                bin_width_sec = time_step * 60
                jitter = rng.uniform(0.0, bin_width_sec, size=len(t_shifted))
                t_shifted = t_shifted + jitter
                t_shifted = np.clip(t_shifted, 0.0, T_total)
                t_shifted.sort()
                
            res = extract_cluster_parameters(
                t_shifted, 
                method=method,
                eps=eps, 
                min_samples=min_samples,
                target_size=target_size
            )
            
            # Save a sample visualization for the first day of each station-day_type
            if f"{sc}_{day_type}" not in plotted_samples:
                plot_sample_clustering(
                    t_shifted, res['centroids'], res['noise'], res['sizes'], 
                    sc, day_type, dt_sec, out_dir_path
                )
                plotted_samples.add(f"{sc}_{day_type}")
            
            key = (sc, day_type)
            if key not in collections:
                collections[key] = {
                    'centroids': [],
                    'sizes': [],
                    'dispersions': [],
                    'noise': [],
                    'num_days': 0
                }
                
            collections[key]['centroids'].extend(res['centroids'])
            collections[key]['sizes'].extend(res['sizes'])
            collections[key]['dispersions'].extend(res['dispersions'])
            collections[key]['noise'].extend(res['noise'])
            collections[key]['num_days'] += 1

    print("\n📈 Aggregating empirical distributions...")
    results = []
    
    for (sc, day_type), col in collections.items():
        if col['num_days'] == 0:
            continue
            
        num_days = col['num_days']
        
        centroids = np.array(col['centroids'])
        noise = np.array(col['noise'])
        sizes = np.array(col['sizes'])
        dispersions = np.array(col['dispersions'])
        
        # 1. Centroid NHPP intensity per bin
        centroid_mu_blocks = np.zeros(num_blocks)
        if len(centroids) > 0:
            bin_indices = (centroids // dt_sec).astype(int)
            bin_indices = np.clip(bin_indices, 0, num_blocks - 1)
            counts = np.bincount(bin_indices, minlength=num_blocks)
            centroid_mu_blocks = (counts / num_days) / dt_sec  # rate per second
            
        # 2. Noise NHPP intensity per bin
        noise_mu_blocks = np.zeros(num_blocks)
        if len(noise) > 0:
            bin_indices = (noise // dt_sec).astype(int)
            bin_indices = np.clip(bin_indices, 0, num_blocks - 1)
            counts = np.bincount(bin_indices, minlength=num_blocks)
            noise_mu_blocks = (counts / num_days) / dt_sec
            
        # 3. Mean cluster size
        cluster_size_mean = float(np.mean(sizes)) if len(sizes) > 0 else 0.0
        
        # 4. Dispersion std
        dispersion_std = float(np.std(dispersions)) if len(dispersions) > 0 else 0.0
        
        res_dict = {
            "station_code": sc,
            "day_type": day_type,
            "num_days": num_days,
            "total_clusters": len(sizes),
            "cluster_size_mean": cluster_size_mean,
            "dispersion_std": dispersion_std,
            "method": method,
        }
        
        # Store blocks in wide format or stringified list.
        # Stringified list is easier for CSV.
        res_dict["centroid_mu_blocks"] = json.dumps(centroid_mu_blocks.tolist())
        res_dict["noise_mu_blocks"] = json.dumps(noise_mu_blocks.tolist())
        
        # For compatibility with UI loaders (is_selected flag if needed)
        res_dict["is_selected"] = True
        
        results.append(res_dict)

    # Save results
    results_df = pd.DataFrame(results)
    out_dir = Path(output_dir) if output_dir else Path("src/workflow/results/cluster_fit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"cluster_params_{count_type}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Completed cluster fitting. Saved {len(results_df)} station-day records to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="src/workflow/params.json")
    parser.add_argument("--stations", type=str, nargs="+", help="Optional list of station codes to analyze")
    parser.add_argument("--date_percentage", type=float, default=1.0, help="Fraction of dates to sample for faster testing")
    parser.add_argument("--count_type", default="checkins", choices=["checkins", "checkouts"])
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="YYYY-MM-DD cutoff for training data (filters to dates <= cutoff)",
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["dbscan", "dbscan_hybrid", "kmeans", "fixed_size"], 
        default="dbscan",
        help="Clustering algorithm to use"
    )
    parser.add_argument(
        "--target_size", 
        type=int, 
        default=5,
        help="Target cluster size for kmeans and fixed_size methods"
    )
    parser.add_argument("--show_plots", action="store_true", help="Show plots interactively")
    args = parser.parse_args()
    run_cluster_fit(
        Path(args.params),
        station_codes_arg=args.stations,
        date_percentage=args.date_percentage,
        count_type=args.count_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        cutoff_date=args.cutoff_date,
        show_plots=args.show_plots,
        method=args.method,
        target_size=args.target_size,
    )
