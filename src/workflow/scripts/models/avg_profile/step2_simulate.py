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
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Simulates days using the fitted profiles.
    """
    if params_path is None:
        params_path = Path("src/workflow/params.json")
    
    with open(params_path) as f:
        params = json.load(f)
    
    avg_params = params.get("avg_profile", {})
    if output_dir is None:
        output_dir = Path(avg_params.get("output_dir", "src/workflow/results/avg_profile"))
    
    params_file = output_dir / f"avg_profile_params_{count_type}.csv"
    if not params_file.exists():
        raise FileNotFoundError(f"Params file not found: {params_file}. Run step1_fit.py first.")
    
    df_params = pd.read_csv(params_file)
    time_cols = [c.replace("_mean", "") for c in df_params.columns if c.endswith("_mean")]
    
    all_simulations = []
    
    print(f"Simulating {n_days} days per station/day_type using {dist_type}...")
    
    for _, row in df_params.iterrows():
        station_code = row["station_code"]
        day_type = row["day_type"]
        
        means = np.array([row[f"{c}_mean"] for c in time_cols])
        stds = np.array([row[f"{c}_std"] for c in time_cols])
        
        for d in range(n_days):
            counts = simulate_day(means, stds, dist_type=dist_type)
            
            sim_row = {
                "station_code": station_code,
                "day_type": day_type,
                "sim_day": d,
                "dist_type": dist_type
            }
            for i, col in enumerate(time_cols):
                sim_row[col] = counts[i]
            
            all_simulations.append(sim_row)
            
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
    
    args = parser.parse_args()
    run_avg_profile_simulate(
        count_type=args.count_type, 
        dist_type=args.dist_type, 
        n_days=args.n_days
    )
