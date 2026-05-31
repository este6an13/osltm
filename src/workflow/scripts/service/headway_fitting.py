"""
Bus Arrival Headway Fitting Script (Service Rate Characterization)

Extracts planned route frequencies, simulates a Transit Delay Mixture Model
representing driver deviations and traffic congestion delays, and fits three 
candidate continuous probability distributions (Gamma, Erlang, Log-Normal) via MLE.
Outputs AIC/BIC comparison metrics to justify service times statistically.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Filter warnings for cleaner console output
warnings.filterwarnings("ignore")


def parse_frequency_string(freq_str: str) -> float:
    """
    Parse strings like '10 min', '4 min', '---' into float minutes.
    Falls back to a default of 5.0 minutes if unparsable or empty.
    """
    if not isinstance(freq_str, str) or "---" in freq_str or not freq_str.strip():
        return 5.0
    
    clean = freq_str.lower().replace("min", "").replace(" ", "").strip()
    try:
        return float(clean)
    except ValueError:
        return 5.0


def simulate_transit_delays(mu_H: float, cv: float, n_samples: int = 1000) -> np.ndarray:
    """
    Generates a realistic Transit Delay Mixture Model:
      H = max(0.1, scheduled_headway + driver_wobble + traffic_congestion_delay)
    
    where:
      - driver_wobble ~ Normal(0, sigma_w^2) represents schedule deviations
      - traffic_congestion_delay ~ Exponential(theta_d) represents traffic delays
    """
    # Wobble is a small fraction of scheduled mean (e.g. 10%)
    sigma_w = 0.10 * mu_H
    
    # Exponential traffic delay represents the rest of the target variation
    # Total variance Var(H) = Var(w) + Var(d) = sigma_w^2 + theta_d^2
    # We want target standard deviation to be cv * mu_H
    target_var = (cv * mu_H) ** 2
    wobble_var = sigma_w ** 2
    
    if target_var > wobble_var:
        theta_d = np.sqrt(target_var - wobble_var)
    else:
        theta_d = 0.05 * mu_H  # tiny fallback delay if cv is extremely small

    # Draw samples
    np.random.seed(42)  # For reproducibility
    w = np.random.normal(0.0, sigma_w, n_samples)
    d = np.random.exponential(theta_d, n_samples)
    
    H = mu_H + w + d
    
    # Clip at 0.1 minutes (6 seconds) to prevent non-positive arrivals
    H = np.clip(H, 0.1, None)
    
    return H


def fit_distributions(H: np.ndarray) -> dict:
    """
    Fit Gamma, Erlang, and Log-Normal distributions to the data using MLE.
    Returns optimal parameters, Log-Likelihood, AIC, and BIC.
    """
    n = len(H)
    results = {}
    
    # Fix loc=0 for all fits to keep support strictly positive (physically correct)
    
    # ── 1. Gamma Fit ───────────────────────────────────────────────────────
    # shape (k), loc, scale (theta)
    k_gam, _, theta_gam = stats.gamma.fit(H, floc=0)
    log_lik_gam = np.sum(stats.gamma.logpdf(H, k_gam, scale=theta_gam))
    aic_gam = 2 * 2 - 2 * log_lik_gam
    bic_gam = np.log(n) * 2 - 2 * log_lik_gam
    
    results["gamma"] = {
        "params": {"shape": float(k_gam), "scale": float(theta_gam)},
        "log_likelihood": float(log_lik_gam),
        "aic": float(aic_gam),
        "bic": float(bic_gam),
    }

    # ── 2. Erlang Fit ──────────────────────────────────────────────────────
    # Erlang is Gamma with integer shape k. We round the Gamma shape and re-fit scale.
    k_erl = max(1, int(round(k_gam)))
    # For a fixed shape k, the MLE scale is simply: mean / k
    theta_erl = np.mean(H) / k_erl
    log_lik_erl = np.sum(stats.erlang.logpdf(H, k_erl, scale=theta_erl))
    aic_erl = 2 * 2 - 2 * log_lik_erl
    # Note: k_erl is technically discrete, but counts as a parameter
    bic_erl = np.log(n) * 2 - 2 * log_lik_erl
    
    results["erlang"] = {
        "params": {"shape_k": int(k_erl), "scale": float(theta_erl)},
        "log_likelihood": float(log_lik_erl),
        "aic": float(aic_erl),
        "bic": float(bic_erl),
    }

    # ── 3. Log-Normal Fit ──────────────────────────────────────────────────
    # s (sigma), loc, scale (exp(mu))
    s_logn, _, scale_logn = stats.lognorm.fit(H, floc=0)
    mu_logn = np.log(scale_logn)
    log_lik_logn = np.sum(stats.lognorm.logpdf(H, s_logn, scale=scale_logn))
    aic_logn = 2 * 2 - 2 * log_lik_logn
    bic_logn = np.log(n) * 2 - 2 * log_lik_logn
    
    results["lognormal"] = {
        "params": {"sigma": float(s_logn), "mu": float(mu_logn), "scale": float(scale_logn)},
        "log_likelihood": float(log_lik_logn),
        "aic": float(aic_logn),
        "bic": float(bic_logn),
    }
    
    return results


def generate_comparison_plot(
    H: np.ndarray,
    fits: dict,
    route_name: str,
    period: str,
    output_path: Path
) -> None:
    """Plot headway histogram with overlaid probability density function curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Histogram of raw simulated headways
    count, bins, _ = ax.hist(H, bins=35, density=True, color="#d1d5db", edgecolor="#9ca3af", alpha=0.6, label="Simulated Headways")
    
    # Generate smooth x-axis grid for continuous plotting
    x = np.linspace(0.1, np.max(H) * 1.1, 300)
    
    # 2. Plot Gamma
    gam = fits["gamma"]["params"]
    ax.plot(x, stats.gamma.pdf(x, gam["shape"], scale=gam["scale"]), color="#1e3a8a", linewidth=2.5, label="Fitted Gamma")
    
    # 3. Plot Erlang
    erl = fits["erlang"]["params"]
    ax.plot(x, stats.erlang.pdf(x, erl["shape_k"], scale=erl["scale"]), color="#15803d", linewidth=2.5, linestyle="--", label=f"Fitted Erlang (k={erl['shape_k']})")
    
    # 4. Plot Log-Normal
    logn = fits["lognormal"]["params"]
    ax.plot(x, stats.lognorm.pdf(x, logn["sigma"], scale=logn["scale"]), color="#b91c1c", linewidth=2.5, linestyle=":", label="Fitted Log-Normal")
    
    ax.set_xlabel("Bus Headway Interval (Minutes)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(f"Headway Distribution Fitting: Route {route_name} ({period.capitalize()} Period)\nScheduled Interval: {np.mean(H):.2f} min", fontsize=13, fontweight="bold")
    
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOTTED] Static comparison plot saved to: {output_path.name}")


def run_headway_fitting(
    route_name: list[str] | str,
    period: str = "peak",
    cv: float = 0.25,
    n_samples: int = 1000,
    output_dir: Optional[Path] = None,
    frequencies_csv: Optional[Path] = None,
) -> dict:
    """Simulates bus arrivals, performs MLE fits, and generates reports."""
    # ── 1. Load Scheduled Frequency ───────────────────────────────────────
    if frequencies_csv is None:
        frequencies_csv = Path("data/routes/transmilenio_frequencies.csv")
    
    if not frequencies_csv.exists():
        # Fallback lookups
        possible_paths = [
            Path("osltm/data/routes/transmilenio_frequencies.csv"),
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "routes" / "transmilenio_frequencies.csv",
            Path("d:/dequi/repositories/osltm/data/routes/transmilenio_frequencies.csv"),
        ]
        for p in possible_paths:
            if p.exists():
                frequencies_csv = p
                break

    if not frequencies_csv.exists():
        raise FileNotFoundError(f"Scheduled frequencies CSV not found at: {frequencies_csv.resolve()}")

    df = pd.read_csv(frequencies_csv)
    
    if isinstance(route_name, str):
        routes = [route_name]
    else:
        routes = list(route_name)

    if output_dir is None:
        output_dir = Path("src/workflow/results/headway_fitting")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    
    for r in routes:
        print(f"\n>>> Processing Route {r}...")
        # Search for matching name (case-insensitive)
        match_df = df[df["name"].str.strip().str.upper() == r.strip().upper()]
        
        if match_df.empty:
            print(f"[WARNING] Route '{r}' not found in frequencies database. Falling back to default scheduled interval of 6.0 minutes.")
            mu_H = 6.0
        else:
            # Load peak or off-peak frequency string
            col = "frequency_peak" if period.lower() == "peak" else "frequency_offpeak"
            freq_str = match_df.iloc[0][col]
            mu_H = parse_frequency_string(freq_str)
            print(f"[LOADED] Loaded scheduled frequency for Route {r} ({period}): {freq_str} -> {mu_H:.1f} minutes")

        # ── 2. Simulate Transit Delays (Proxy Telemetry) ──────────────────────
        print(f"[SIMULATING] Simulating {n_samples} headways from Transit Delay Mixture Model (Cv={cv:.2f})...")
        H = simulate_transit_delays(mu_H, cv, n_samples)
        
        # Save simulated samples to CSV so the results viewer/API can read it
        samples_file = output_dir / f"simulated_headways_{r}.csv"
        pd.DataFrame({"headway_minutes": H}).to_csv(samples_file, index=False)
        print(f"[SAVED] Raw simulated samples saved to: {samples_file}")

        # Save single fallback file for backwards compatibility
        if r == routes[0] or len(routes) == 1:
            sh_file = output_dir / "simulated_headways.csv"
            pd.DataFrame({"headway_minutes": H}).to_csv(sh_file, index=False)

        # ── 3. Perform MLE fits ────────────────────────────────────────────────
        print("[FITTING] Fitting Gamma, Erlang, and Log-Normal models via scipy MLE...")
        fits = fit_distributions(H)
        
        # Save fits to JSON
        meta_file = output_dir / f"fitted_headway_report_{r}.json"
        report_data = {
            "route_name": r,
            "period": period,
            "cv": cv,
            "scheduled_mean": mu_H,
            "simulated_mean": float(np.mean(H)),
            "simulated_std": float(np.std(H)),
            "fits": fits
        }
        with open(meta_file, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"[SAVED] Goodness-of-fit parameters report exported to: {meta_file}")

        if r == routes[0] or len(routes) == 1:
            default_meta_file = output_dir / "fitted_headway_report.json"
            with open(default_meta_file, "w") as f:
                json.dump(report_data, f, indent=2)

        # ── 4. Generate Plot ──────────────────────────────────────────────────
        plot_file = output_dir / f"fitted_headways_comparison_{r}.png"
        generate_comparison_plot(H, fits, r, period, plot_file)
        
        if r == routes[0] or len(routes) == 1:
            default_plot_file = output_dir / "fitted_headways_comparison.png"
            generate_comparison_plot(H, fits, r, period, default_plot_file)
        
        # Print summary table
        print(f"\n[RESULTS] {r} GOODNESS-OF-FIT CRITERIA:")
        print(f"{'Distribution':<15} | {'Log-Likelihood':<15} | {'AIC':<12} | {'BIC':<12}")
        print("-" * 62)
        for dist, dat in fits.items():
            print(f"{dist.capitalize():<15} | {dat['log_likelihood']:<15.3f} | {dat['aic']:<12.3f} | {dat['bic']:<12.3f}")
        
        # Determine winner
        winner = min(fits.keys(), key=lambda k: fits[k]["aic"])
        print(f"\n[WINNER] Best-fitting distribution (by AIC/BIC): {winner.upper()}")

        results[r] = {
            "samples": H,
            "fits": fits,
            "winner": winner
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stochastically perturbs planned bus frequencies and fits Gamma, Erlang, and Log-Normal models."
    )
    parser.add_argument(
        "--route_name",
        type=str,
        nargs="+",
        default=["B12"],
        help="Transit service name(s) to model (default: B12)",
    )
    parser.add_argument(
        "--period",
        choices=["peak", "offpeak"],
        default="peak",
        help="Day period to model (default: peak)",
    )
    parser.add_argument(
        "--cv",
        type=float,
        default=0.25,
        help="Stochastic traffic delay Coefficient of Variation (default: 0.25)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of headway samples to draw (default: 1000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/headway_fitting",
        help="Output folder (default: src/workflow/results/headway_fitting)",
    )
    parser.add_argument(
        "--frequencies",
        type=str,
        default="data/routes/transmilenio_frequencies.csv",
        help="Path to route frequencies database CSV",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="src/workflow/params.json",
        help="Path to params.json (ignored but accepted for runner compatibility)",
    )

    args = parser.parse_args()
    
    run_headway_fitting(
        route_name=args.route_name,
        period=args.period,
        cv=args.cv,
        n_samples=args.n_samples,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        frequencies_csv=Path(args.frequencies) if args.frequencies else None,
    )
    
    print("\n[FINISHED] Week 2 Bus Headway Stochastic Fitting successfully finished!")
