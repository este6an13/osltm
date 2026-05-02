r"""
Fits the continuous-time Hawkes process to exact event timestamps (check-ins or check-outs).
Uses the 15-minute aggregate profiles as the baseline intensity $\mu_{base}(t)$.

Note on checkouts:
    Checkout data is recorded as aggregate counts per 15-minute bin (Salidas_S),
    not as individual event timestamps.  To enable Hawkes fitting, each count is
    expanded into individual events and a uniform jitter of U(0, bin_width) is
    added to spread events within their bin.  This is a pseudo-continuous
    approximation — it does NOT recover the real inter-event structure, and the
    fitted excitation parameters should be interpreted with caution.  The LGCP
    pipeline (which models binned counts directly) is the recommended approach
    for checkout intensity analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.workflow.data_loader import load_data, load_persisted_data
from src.workflow.scripts.models.hawkes.core import fit_hawkes, compute_compensator_tau
from src.workflow.data_reader import load_csv_file

def compute_mu_base_profile(station_df: pd.DataFrame, time_min: int, time_max: int, time_step: int):
    """
    Computes the normalized background intensity piecewise function 
    for each day_type based on the historical 15-minute means.
    
    Returns:
        dict mapping day_type to (mu_blocks, M_blocks, T_total)
        where mu_blocks is the array of constant rates (per second) 
        and M_blocks is cumulative integral.
    """
    # Create time boundaries in seconds
    # Example 400 to 2300.
    # Convert time_min (e.g. 400) to seconds since midnight
    start_sec = (time_min // 100) * 3600 + (time_min % 100) * 60
    end_sec = (time_max // 100) * 3600 + (time_max % 100) * 60
    
    # We will measure t relative to start_sec, so T_total = end_sec - start_sec
    T_total = end_sec - start_sec
    dt_sec = time_step * 60
    num_blocks = T_total // dt_sec
    
    time_cols = [col for col in station_df.columns if col.startswith("t_")]
    # sort time_cols to be safe
    time_cols.sort(key=lambda x: int(x.split("_")[1]))
    
    profiles = {}
    day_types = station_df["date_type"].unique()
    
    for dt in day_types:
        dt_df = station_df[station_df["date_type"] == dt]
        means = dt_df[time_cols].mean(axis=0).values
        
        # normalize
        total_mean = np.sum(means)
        if total_mean <= 0:
            continue
            
        proportions = means / total_mean
        # mu in units of (probability/second)
        mu_blocks = proportions / dt_sec
        # cumulative integral at the boundaries of the blocks
        M_blocks = np.concatenate([[0.0], np.cumsum(proportions)])
        
        profiles[dt] = {
            "mu_blocks": mu_blocks,
            "M_blocks": M_blocks,
            "dt_sec": dt_sec,
            "T_total": T_total,
            "start_sec": start_sec
        }
        
    return profiles

def get_mu_values_for_timestamps(t, profile):
    """
    Given an array of timestamps t (in seconds from start_sec)
    and the profile parameter dict, return mu_base(t) and M_base(t).
    """
    dt_sec = profile["dt_sec"]
    mu_blocks = profile["mu_blocks"]
    M_blocks = profile["M_blocks"]
    
    # integer division to find the block
    idx = (t // dt_sec).astype(int)
    # Clip to valid indices just in case of rounding
    idx = np.clip(idx, 0, len(mu_blocks) - 1)
    
    mu_t = mu_blocks[idx]
    
    # Fractional part within the block
    frac = t - (idx * dt_sec)
    M_t = M_blocks[idx] + mu_blocks[idx] * frac
    
    return mu_t, M_t

def p_value_from_tau(tau):
    """
    Computes KS test p-value for exponential inter-arrival times of tau.
    """
    if len(tau) < 2:
        return np.nan
        
    diffs = np.diff(tau)
    # They should be Exp(1).
    # Provide exact CDF of Exp(1): F(x) = 1 - exp(-x)
    # KS test transforms diffs to Uniform(0,1) with F(x) and tests vs Uniform
    res = stats.kstest(diffs, 'expon', args=(0, 1))
    return res.pvalue

def run_hawkes_fit(
    params_path: Path,
    station_codes_arg: Optional[list[str]] = None,
    date_percentage: float = 1.0,
    count_type: str = "checkins",
    seed: int = 42,
    output_dir: Optional[Path] = None,
    cutoff_date: Optional[str] = None,
):
    with open(params_path) as f:
        params = json.load(f)
        
    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)
    
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
        # Use provided stations, preferably if they are sampled
        station_codes = [sc for sc in station_codes_arg if sc in available_station_codes]
        if not station_codes:
            print("⚠️ Provided stations were not in the sampled stations. Proceeding anyway.")
            station_codes = station_codes_arg
    else:
        station_codes = available_station_codes
    
    # 1. Load 15-min data to get the base profiles
    print(f"📊 Loading aggregated {count_type} data to build baseline profiles...")
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
    
    # Map from station_code -> profile dictionary
    station_profiles = {}
    for sc in station_codes:
        st_df = df_15min[df_15min["station_code"] == sc]
        if not st_df.empty:
            station_profiles[sc] = compute_mu_base_profile(st_df, time_min, time_max, time_step)
            
    # Also create a fast lookup for date -> date_type from df_15min
    date_to_daytype = {}
    for _, row in df_15min[["year", "month", "day", "date_type"]].drop_duplicates().iterrows():
        # format string YYYYMMDD
        date_str = f"{row['year']:04d}{row['month']:02d}{row['day']:02d}"
        date_to_daytype[date_str] = row['date_type']

    results = []
    rng = np.random.default_rng(seed)
    
    # 2. Iterate through daily raw CSVs
    raw_dir = (
        Path("data/check_ins/daily")
        if count_type == "checkins"
        else Path("data/check_outs/daily")
    )

    is_checkout = count_type == "checkouts"
    if is_checkout:
        print(
            "⚠️  Checkout data is 15-min aggregated.  Applying uniform jitter "
            "U(0, bin_width) to spread events within each bin.\n"
            "    This is a pseudo-continuous approximation — fitted excitation "
            "parameters may NOT reflect real self-excitation structure."
        )

    print(f"🔍 Fitting Hawkes processes to {'jittered ' if is_checkout else ''}{count_type} timestamps...")
    for date_str in sampled_dates:
        csv_path = raw_dir / f"{date_str}.csv"
        if not csv_path.exists():
            print(f"⚠️ Missing raw file: {csv_path}")
            continue
            
        day_type = date_to_daytype.get(date_str, "WD") # fallback
        
        print(f"   Processing {date_str} ({day_type})...")
        try:
            df_raw = load_csv_file(
                csv_path,
                station_codes=station_codes,
                count_type=count_type,
                include_time_components=True,
            )
            # Compute seconds-from-midnight for each event
            df_raw["sec"] = (
                df_raw["hour"] * 3600
                + df_raw["minute"] * 60
                + df_raw["second"]
            )
        except Exception as e:
            print(f"❌ Failed to read {csv_path}: {e}")
            continue
            
        for sc in station_codes:
            if sc not in station_profiles or day_type not in station_profiles[sc]:
                continue
                
            profile = station_profiles[sc][day_type]
            
            # Filter rows for this station (already filtered by load_csv_file, but check anyway)
            st_data = df_raw[df_raw["station_code"] == sc].copy()
            
            if len(st_data) < 10:
                print(f"      {sc} {date_str} skipped: not enough data ({len(st_data)})")
                continue # not enough data to fit
                
            # Filter by time bounds
            t_mask = (st_data["sec"] >= profile["start_sec"]) & (st_data["sec"] <= profile["start_sec"] + profile["T_total"])
            t_sec = st_data.loc[t_mask, "sec"].sort_values().values
            
            # shift to start at 0
            t_shifted = t_sec - profile["start_sec"]
            T_total = profile["T_total"]

            # ── Checkout jitter ────────────────────────────────────────────
            # Checkout timestamps sit on 15-min boundaries (all events in a
            # bin share the exact same second).  Add U(0, bin_width) jitter
            # to produce a pseudo-continuous stream.  This does NOT recover
            # the true inter-arrival structure — it merely prevents the
            # degenerate dt=0 clusters that otherwise cause the optimizer
            # to push α,β to extreme values.
            if is_checkout:
                bin_width_sec = profile["dt_sec"]  # 900 s for 15-min bins
                jitter = rng.uniform(0.0, bin_width_sec, size=len(t_shifted))
                t_shifted = t_shifted + jitter
                # Clamp to [0, T_total] and re-sort
                t_shifted = np.clip(t_shifted, 0.0, T_total)
                t_shifted.sort()
            
            mu_t, M_t = get_mu_values_for_timestamps(t_shifted, profile)
            
            # If mu_t has exact zeros (empirical bins with 0 counts where events actually occurred),
            # the Hawkes log-likelihood evaluates to log(0), causing the optimizer to fail instantly.
            # We add a tiny epsilon to ensure strict positivity.
            mu_t = np.maximum(mu_t, 1e-6)

            # Fit Hawkes
            try:
                fit_res = fit_hawkes(t_shifted, mu_t, 1.0, T_total)
            except Exception as e:
                # the numeric objective function threw hard failing exceptions
                print(f"      {sc} {date_str} math exception: {e}")
                continue
            
            if not fit_res['converged']:
                print(f"      {sc} {date_str} WARNING: optimizer did not converge ({fit_res['message']}), but keeping parameters.")
                
            params = [fit_res['kappa'], fit_res['alpha'], fit_res['beta']]
            
            # Goodness-of-Fit via time rescaling
            tau = compute_compensator_tau(params, t_shifted, M_t)
            p_val = p_value_from_tau(tau)
            
            results.append({
                "date": date_str,
                "day_type": day_type,
                "station_code": sc,
                "n_arrivals": len(t_sec),
                "kappa": fit_res['kappa'],
                "alpha": fit_res['alpha'],
                "beta": fit_res['beta'],
                "branching_ratio": fit_res['branching_ratio'],
                "loglik": fit_res['loglik'],
                "gof_ks_pval": p_val
            })
            
    # Save results
    results_df = pd.DataFrame(results)
    out_dir = Path(output_dir) if output_dir else Path("src/workflow/results/hawkes_fit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"hawkes_params_{count_type}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Completed Hawkes process fits. Saved {len(results_df)} station-day records to {out_csv}")
    
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
    args = parser.parse_args()
    run_hawkes_fit(
        Path(args.params),
        station_codes_arg=args.stations,
        date_percentage=args.date_percentage,
        count_type=args.count_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        cutoff_date=args.cutoff_date,
    )
