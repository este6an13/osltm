"""
Diagnostics suite for the Hawkes Process fitting results.
Generates metrics and plots to validate the branching ratios and goodness-of-fit.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summarize_results(params_path: Path):
    if not params_path.exists():
        print(f"❌ File not found: {params_path}")
        return
        
    df = pd.read_csv(params_path)
    
    print("\n" + "="*50)
    print("HAWKES PROCESS DIAGNOSTICS")
    print("="*50)
    print(f"Total Converged Fits: {len(df)}")
    
    if len(df) == 0:
        return
        
    # Stats by Day Type
    print("\n--- By Day Type ---")
    summary = df.groupby("day_type").agg(
        n_fits=('date', 'size'),
        mean_branching_ratio=('branching_ratio', 'mean'),
        median_branching_ratio=('branching_ratio', 'median'),
        mean_alpha=('alpha', 'mean'),
        mean_beta=('beta', 'mean'),
        pct_valid_gof=('gof_ks_pval', lambda x: (x > 0.05).mean() * 100)
    ).round(4)
    print(summary.to_string())
    
    return df

def generate_plots(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Branching Ratio Boxplots by Day Type
    plt.figure(figsize=(8, 6))
    
    day_types = df["day_type"].unique()
    data_to_plot = [df[df["day_type"] == dt]["branching_ratio"].dropna().values for dt in day_types]
    
    plt.boxplot(data_to_plot, labels=day_types)
    plt.axhline(1.0, color='r', linestyle='--', label='Critical Threshold ($n=1$)')
    plt.axhline(0.0, color='k', linestyle='-')
    plt.ylabel("Branching Ratio ($n = \\alpha / \\beta$)")
    plt.title("Distributions of Hawkes Branching Ratios")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(output_dir / "hawkes_branching_ratio_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: KS Test p-value histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df["gof_ks_pval"].dropna(), bins=20, color='skyblue', edgecolor='black')
    plt.axvline(0.05, color='red', linestyle='dashed', linewidth=2, label='0.05 (significance)')
    plt.xlabel("KS Test p-value (Uniformity of Rescaled Times)")
    plt.ylabel("Frequency")
    plt.title("Goodness-of-Fit Distribution (Time-Rescaling Theorem)")
    plt.legend()
    
    plt.savefig(output_dir / "hawkes_ks_pval_hist.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n✅ Diagnostic plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count_type", default="checkins", choices=["checkins", "checkouts"])
    parser.add_argument(
        "--input",
        default=None,
        help="Path to hawkes_params CSV. Defaults to results/hawkes_fit/hawkes_params_{count_type}.csv",
    )
    parser.add_argument("--output_dir", default="src/workflow/results/hawkes_fit")
    args = parser.parse_args()

    input_path = (
        Path(args.input)
        if args.input
        else Path(f"src/workflow/results/hawkes_fit/hawkes_params_{args.count_type}.csv")
    )
    df = summarize_results(input_path)
    if df is not None:
        generate_plots(df, Path(args.output_dir))

if __name__ == "__main__":
    main()
