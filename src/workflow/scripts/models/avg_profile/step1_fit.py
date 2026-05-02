"""
Average Profile Model - Step 1: Fitting
Calculates mean and standard deviation profiles per (station, day_type) 
using data up to a specified cutoff date.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import numpy as np

from src.workflow.data_loader import load_data, load_persisted_data

def run_avg_profile_fit(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    cutoff_date: Optional[str] = None,
    min_days: int = 5,
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Fits the average profile model by computing mean and std per station/day_type.
    """
    if params_path is None:
        params_path = Path("src/workflow/params.json")
    
    with open(params_path) as f:
        params = json.load(f)
    
    avg_params = params.get("avg_profile", {})
    if cutoff_date is None:
        cutoff_date = avg_params.get("cutoff_date")
    if output_dir is None:
        output_dir = Path(avg_params.get("output_dir", "src/workflow/results/avg_profile"))
    else:
        output_dir = Path(output_dir)
    if min_days is None:
        min_days = avg_params.get("min_days", 5)

    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)

    persistence_dir = params_path.parent
    _, sampled_stations = load_persisted_data(persistence_dir)
    
    if not station_codes and sampled_stations:
        station_codes = [s["code"] for s in sampled_stations]

    print(f"Loading {count_type} data for training (up to {cutoff_date})...")
    
    # We load ALL data and filter here to handle the cutoff logic simply
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min, time_max=time_max, time_step=time_step,
    )
    
    df = data[count_type]
    
    # Filter by day types if provided
    if day_types and len(day_types) > 0:
        df = df[df["date_type"].isin(day_types)].copy()
    
    # Create a proper date column for filtering
    df["date_str"] = df.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
    
    # Filter training data
    train_df = df[df["date_str"] <= cutoff_date].copy()
    print(f"   Training records: {len(train_df)} (from total {len(df)})")

    time_cols = [c for c in train_df.columns if c.startswith("t_")]
    
    results = []
    
    grouped = train_df.groupby(["station_code", "date_type"])
    for (station_code, day_type), group in grouped:
        if len(group) < min_days:
            continue
            
        means = group[time_cols].mean().fillna(0)
        stds = group[time_cols].std().fillna(0)
        
        res = {
            "station_code": station_code,
            "day_type": day_type,
            "n_days": len(group)
        }
        for col in time_cols:
            res[f"{col}_mean"] = means[col]
            res[f"{col}_std"] = stds[col]
            
        results.append(res)

    results_df = pd.DataFrame(results)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"avg_profile_params_{count_type}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(results_df)} station/day_type profiles to {output_file}")
    return results_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit Average Profile Baseline Model")
    parser.add_argument("--count_type", choices=["checkins", "checkouts"], default="checkins")
    parser.add_argument("--cutoff_date", type=str, help="YYYY-MM-DD cutoff for training")
    parser.add_argument("--params", type=str, help="Path to params.json")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--stations", nargs="+", help="Subset of station codes")
    parser.add_argument("--day_types", nargs="+", help="Subset of day types (WD, SA, SU, HO)")
    
    args = parser.parse_args()
    run_avg_profile_fit(
        count_type=args.count_type, 
        cutoff_date=args.cutoff_date,
        params_path=Path(args.params) if args.params else None,
        output_dir=args.output_dir,
        station_codes=args.stations,
        day_types=args.day_types
    )
