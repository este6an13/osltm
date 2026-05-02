"""
Average Profile Model - Step 2: Simulation
Generates synthetic days using Poisson or Negative Binomial distributions
based on fitted average profiles.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import numpy as np
from scipy import stats

def simulate_day(means: np.ndarray, stds: np.ndarray, dist_type: str = "poisson") -> np.ndarray:
    """Simulate one day of counts bin-wise."""
    if dist_type == "poisson":
        return np.random.poisson(means)
    elif dist_type == "neg_binomial":
        # NB(n, p) has mean n(1-p)/p and var n(1-p)/p^2
        # p = mean / var
        # n = mean^2 / (var - mean)
        variances = stds ** 2
        simulated = np.zeros_like(means)
        
        for i in range(len(means)):
            m = means[i]
            v = variances[i]
            
            if m <= 0:
                simulated[i] = 0
            elif v <= m:
                # Fallback to Poisson if underdispersed or exactly Poisson
                simulated[i] = np.random.poisson(m)
            else:
                p = m / v
                n = (m ** 2) / (v - m)
                simulated[i] = stats.nbinom.rvs(n, p)
        return simulated
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

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
    Simulates days based on the fitted average profiles.
    """
    if params_path is None:
        params_path = Path("src/workflow/params.json")
        
    with open(params_path) as f:
        params = json.load(f)
        
    avg_params = params.get("avg_profile", {})
    if output_dir is None:
        output_dir = Path(avg_params.get("output_dir", "src/workflow/results/avg_profile"))
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parameters from fit_dir if provided, else from output_dir
    if fit_dir is None:
        fit_dir = output_dir
    else:
        fit_dir = Path(fit_dir)

    param_file = fit_dir / f"avg_profile_params_{count_type}.csv"
    if not param_file.exists():
        raise FileNotFoundError(f"Average profile parameters not found: {param_file}. Run fit step first.")
        
    df_params = pd.read_csv(param_file)
    df_params["station_code"] = df_params["station_code"].astype(str).str.zfill(5)
    
    # Filter by stations/day types if provided
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
            
            res = {
                "station_code": station_code,
                "day_type": day_type,
                "sim_day": d
            }
            for i, col in enumerate(time_cols):
                res[col] = counts[i]
                
            all_simulations.append(res)
            
    sim_df = pd.DataFrame(all_simulations)
    output_file = output_dir / f"avg_profile_simulations_{count_type}_{dist_type}.csv"
    sim_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(sim_df)} simulated days to {output_file}")
    return sim_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate Average Profile Baseline Model")
    parser.add_argument("--count_type", choices=["checkins", "checkouts"], default="checkins")
    parser.add_argument("--dist_type", choices=["poisson", "neg_binomial"], default="poisson")
    parser.add_argument("--n_days", type=int, default=30, help="Number of days to simulate per station/day_type")
    parser.add_argument("--params", type=str, help="Path to params.json")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--fit_dir", type=str, help="Directory containing fitted parameters")
    parser.add_argument("--stations", nargs="+", help="Subset of station codes")
    parser.add_argument("--day_types", nargs="+", help="Subset of day types")
    
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
