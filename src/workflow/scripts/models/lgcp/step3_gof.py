"""
LGCP Goodness-of-Fit: PIT-based Model Comparison (Phase 4)

Compares the LGCP (Poisson-LogNormal) against the NHPP (Poisson) using
the Probability Integral Transform (PIT) with randomized quantile residuals
(Dunn & Smyth, 1996).

For each (station, day_type, day, bin):
  - NHPP:  N_k ~ Poisson(λ̂_k)         where λ̂_k = sample mean
  - LGCP:  N_k ~ PoissonLogNormal(μ_k, σ²)
           i.e.  z_k ~ N(μ_k, σ²),  N_k | z_k ~ Poisson(exp(z_k))

Randomized quantile residuals:
  u ~ Uniform(F(n-1), F(n))
Under the correct model, u should be i.i.d. Uniform(0,1).

Outputs:
  - lgcp_gof_pit_comparison_{ct}.csv  — KS stats: Poisson vs PLN
  - lgcp_gof_pit_qq_{s}_{d}.png      — side-by-side PIT QQ-plots
  - lgcp_gof_pit_summary_{ct}.png    — bar chart of KS improvement
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.workflow.data_loader import load_data, load_persisted_data

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]


# ── PIT computation ──────────────────────────────────────────────────────────


def poisson_cdf(n: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Poisson CDF: P(N <= n | λ)."""
    return stats.poisson.cdf(n, lam)


def poisson_lognormal_cdf(
    n: np.ndarray,
    mu: np.ndarray,
    sigma2: float,
    n_mc: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Poisson-LogNormal CDF via Monte Carlo.

    F_PLN(n; μ, σ²) = E_z[ F_Poisson(n; exp(z)) ]  where z ~ N(μ, σ²)

    Args:
        n: observed counts (J, K)
        mu: mean log-intensity per bin (K,)
        sigma2: GP signal variance (scalar)
        n_mc: Monte Carlo samples
        seed: RNG seed

    Returns:
        CDF values, same shape as n
    """
    rng = np.random.default_rng(seed)
    J, K = n.shape
    sigma = np.sqrt(max(sigma2, 1e-10))

    # z_samples: (n_mc, K)
    z_samples = rng.normal(mu[None, :], sigma, size=(n_mc, K))
    lam_samples = np.exp(z_samples)  # (n_mc, K)

    # For each MC sample, compute F_Poisson(n_{j,k}; lam_m_k)
    # n is (J, K), lam_samples is (n_mc, K)
    # We need F(n_{j,k}; lam_{m,k}) averaged over m
    # = (1/M) sum_m F_Poisson(n_{j,k}; lam_{m,k})

    # Vectorize: expand n to (J, 1, K), lam to (1, n_mc, K)
    n_exp = n[:, None, :]        # (J, 1, K)
    lam_exp = lam_samples[None, :, :]  # (1, n_mc, K)

    # F_Poisson for each (j, m, k) — this is the expensive step
    # Process in chunks to avoid memory issues
    chunk_size = 50
    cdf_sum = np.zeros((J, K))
    for i in range(0, n_mc, chunk_size):
        end = min(i + chunk_size, n_mc)
        lam_chunk = lam_exp[:, i:end, :]  # (1, chunk, K)
        cdf_chunk = stats.poisson.cdf(n_exp, lam_chunk)  # (J, chunk, K)
        cdf_sum += cdf_chunk.sum(axis=1)  # (J, K)

    return cdf_sum / n_mc


def randomized_pit(
    n: np.ndarray,
    cdf_at_n: np.ndarray,
    cdf_at_n_minus_1: np.ndarray,
    seed: int = 123,
) -> np.ndarray:
    """
    Randomized quantile residuals (Dunn & Smyth 1996).

    u ~ Uniform(F(n-1), F(n))

    For discrete distributions, F(n) is a step function, so the raw PIT
    has point masses. Randomization makes it continuous.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(cdf_at_n_minus_1, cdf_at_n)
    return u


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_pit_qq_side_by_side(
    u_poisson: np.ndarray,
    u_pln: np.ndarray,
    station_code: str,
    station_name: str,
    day_type: str,
    ks_poisson: float,
    ks_pln: float,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """Side-by-side PIT QQ-plots: Poisson vs PLN."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    dt_label = DATE_TYPE_LABELS.get(day_type, day_type)

    for ax, u_vals, label, ks in [
        (axes[0], u_poisson, "Poisson (NHPP)", ks_poisson),
        (axes[1], u_pln, "Poisson-LogNormal (LGCP)", ks_pln),
    ]:
        if len(u_vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        n = len(u_vals)
        theoretical = np.linspace(0, 1, n + 2)[1:-1]
        sample = np.sort(u_vals)

        ax.scatter(theoretical, sample, alpha=0.15, s=8, color="steelblue")
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect fit")

        p_val = stats.kstest(u_vals, "uniform").pvalue
        ax.text(
            0.05, 0.95,
            f"KS = {ks:.4f}\np = {p_val:.2e}\nn = {n}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=10,
        )

        ax.set_xlabel("Theoretical Quantiles", fontsize=11)
        ax.set_ylabel("PIT Quantiles", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle(
        f"PIT QQ — {station_code} {station_name}\n{dt_label} ({count_type.capitalize()})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"lgcp_gof_pit_qq_{station_code}_{day_type}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  -> {fname.name}")
    plt.close()


def plot_pit_summary(
    comparison_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """Bar chart: KS statistic under Poisson vs PLN."""
    fig, ax = plt.subplots(figsize=(14, 6))

    labels = [
        f"{row['station_code']}\n{DATE_TYPE_LABELS.get(row['day_type'], row['day_type'])}"
        for _, row in comparison_df.iterrows()
    ]
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(
        x - width / 2, comparison_df["ks_poisson"], width,
        label="Poisson (NHPP)", color="#e74c3c", alpha=0.8,
    )
    ax.bar(
        x + width / 2, comparison_df["ks_pln"], width,
        label="Poisson-LogNormal (LGCP)", color="#2980b9", alpha=0.8,
    )

    ax.set_ylabel("KS Statistic", fontsize=12)
    ax.set_xlabel("Station / Day-type", fontsize=12)
    ax.set_title(
        f"PIT Goodness-of-Fit: Poisson vs LGCP ({count_type.capitalize()})",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate reduction %
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        if row["ks_poisson"] > 0:
            reduction = (1 - row["ks_pln"] / row["ks_poisson"]) * 100
            y_pos = max(row["ks_poisson"], row["ks_pln"]) + 0.005
            ax.text(
                i, y_pos, f"{reduction:+.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="green" if reduction > 0 else "red",
            )

    plt.tight_layout()

    if output_dir:
        fname = output_dir / f"lgcp_gof_pit_summary_{count_type}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved PIT summary chart: {fname}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def sort_time_columns(cols: list[str]) -> list[str]:
    return sorted(cols, key=lambda c: int(c.replace("t_", "")))


def run_lgcp_gof(
    count_type: str = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    phase2_dir: Optional[Path] = None,
    min_days: int = 5,
    n_mc: int = 500,
) -> dict:
    """
    Run PIT-based goodness-of-fit: Poisson (NHPP) vs PoissonLogNormal (LGCP).
    """
    # ── Parameters ────────────────────────────────────────────────────────
    if params_path is None:
        params_path = Path("src/workflow/params.json")
    with open(params_path) as f:
        params = json.load(f)

    step4 = params.get("step4", {})
    time_min = step4.get("time_min", 400)
    time_max = step4.get("time_max", 2300)
    time_step = step4.get("time_step", 15)

    persistence_dir = params_path.parent
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)
    if not sampled_dates:
        raise ValueError("No sampled dates. Run workflow steps 1-4 first.")

    if sampled_stations:
        available = [s["code"] for s in sampled_stations]
        station_codes = (
            [sc for sc in station_codes if sc in available]
            if station_codes
            else available
        )
    elif station_codes is None:
        raise ValueError("No station codes available.")

    # ── Load Phase 2 kernel params (σ² per station/day_type) ──────────────
    if phase2_dir is None:
        phase2_dir = Path("src/workflow/results/lgcp_twostage")
    phase2_dir = Path(phase2_dir)

    kernel_csv = phase2_dir / f"lgcp_kernel_params_{count_type}.csv"
    if not kernel_csv.exists():
        raise FileNotFoundError(f"Phase 2 kernel params not found: {kernel_csv}")
    kernel_df = pd.read_csv(kernel_csv)
    kernel_df["station_code"] = kernel_df["station_code"].astype(str).str.zfill(5)
    # Keep only selected kernels
    kernel_sel = kernel_df[kernel_df["is_selected"] == True].copy()
    print(f"Loaded {len(kernel_sel)} selected kernels from Phase 2")

    # ── Load count data ───────────────────────────────────────────────────
    print(f"Loading {count_type} data...")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min, time_max=time_max, time_step=time_step,
    )
    df = data[count_type]
    time_cols = sort_time_columns([c for c in df.columns if c.startswith("t_")])
    K = len(time_cols)

    if output_dir is None:
        output_dir = Path("src/workflow/results/lgcp_gof")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Main loop per (station, day_type) ─────────────────────────────────
    comparison_rows: list[dict] = []

    for _, krow in kernel_sel.iterrows():
        station_code = krow["station_code"]
        day_type = krow["day_type"]
        sigma2 = krow["sigma2"]

        if station_code not in station_codes:
            continue

        dt_label = DATE_TYPE_LABELS.get(day_type, day_type)

        sdf = df[
            (df["station_code"] == station_code) & (df["date_type"] == day_type)
        ]
        J = len(sdf)
        if J < min_days:
            continue

        counts = sdf[time_cols].fillna(0).values.astype(float)  # (J, K)
        mean_counts = counts.mean(axis=0)  # (K,) — NHPP rate estimate

        # Prior mean in log-space
        eps0 = 0.5
        mu = np.log(mean_counts + eps0)  # (K,)

        print(f"\n{station_code}/{dt_label}  J={J}, K={K}, σ²={sigma2:.4f}")

        # ── Poisson PIT ───────────────────────────────────────────────────
        print("  Poisson PIT...")
        lam_nhpp = np.maximum(mean_counts, 1e-6)  # (K,)
        # Broadcast: lam_nhpp is (K,) -> tile to (J, K)
        lam_mat = np.tile(lam_nhpp, (J, 1))

        cdf_pois_n = poisson_cdf(counts, lam_mat)           # F(n)
        cdf_pois_nm1 = poisson_cdf(counts - 1, lam_mat)     # F(n-1)
        u_poisson = randomized_pit(counts, cdf_pois_n, cdf_pois_nm1, seed=42)
        u_poisson_flat = u_poisson.ravel()

        # ── Poisson-LogNormal PIT ─────────────────────────────────────────
        print("  Poisson-LogNormal PIT (MC)...")
        cdf_pln_n = poisson_lognormal_cdf(counts, mu, sigma2, n_mc=n_mc, seed=42)
        cdf_pln_nm1 = poisson_lognormal_cdf(
            np.maximum(counts - 1, 0), mu, sigma2, n_mc=n_mc, seed=42
        )
        u_pln = randomized_pit(counts, cdf_pln_n, cdf_pln_nm1, seed=42)
        u_pln_flat = u_pln.ravel()

        # ── KS tests ─────────────────────────────────────────────────────
        ks_pois, p_pois = stats.kstest(u_poisson_flat, "uniform")
        ks_pln, p_pln = stats.kstest(u_pln_flat, "uniform")
        reduction = (1 - ks_pln / ks_pois) * 100 if ks_pois > 0 else 0.0

        print(f"  Poisson KS={ks_pois:.4f} (p={p_pois:.2e})")
        print(f"  PLN     KS={ks_pln:.4f} (p={p_pln:.2e})")
        print(f"  Improvement: {reduction:+.1f}%")

        # Get station name
        station_name = (
            sdf["station_name"].iloc[0] if "station_name" in sdf.columns
            else station_code
        )

        comparison_rows.append({
            "station_code": station_code,
            "station_name": station_name,
            "day_type": day_type,
            "n_obs": J * K,
            "sigma2": sigma2,
            "ks_poisson": ks_pois,
            "p_poisson": p_pois,
            "ks_pln": ks_pln,
            "p_pln": p_pln,
            "ks_reduction_pct": reduction,
        })

        # ── QQ-plot ───────────────────────────────────────────────────────
        plot_pit_qq_side_by_side(
            u_poisson_flat, u_pln_flat,
            station_code, station_name, day_type,
            ks_pois, ks_pln,
            output_dir=output_dir,
            count_type=count_type,
        )

    # ── Save CSV ──────────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = output_dir / f"lgcp_gof_pit_comparison_{count_type}.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Summary chart ─────────────────────────────────────────────────────
    if not comparison_df.empty:
        plot_pit_summary(comparison_df, output_dir=output_dir, count_type=count_type)

    # ── Summary stats ─────────────────────────────────────────────────────
    print("\n=== Summary ===")
    if not comparison_df.empty:
        valid = comparison_df.dropna(subset=["ks_reduction_pct"])
        if not valid.empty:
            print(f"  Mean KS reduction: {valid['ks_reduction_pct'].mean():+.1f}%")
            print(f"  Median KS reduction: {valid['ks_reduction_pct'].median():+.1f}%")
            n_improved = (valid["ks_reduction_pct"] > 0).sum()
            print(f"  LGCP improved in {n_improved}/{len(valid)} pairs")

    return {"comparison_df": comparison_df}


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PIT-based GoF: Poisson (NHPP) vs Poisson-LogNormal (LGCP)"
    )
    parser.add_argument("--count_type", default="checkins", choices=["checkins", "checkouts"])
    parser.add_argument("--output_dir", default="src/workflow/results/lgcp_gof")
    parser.add_argument("--params", default="src/workflow/params.json")
    parser.add_argument("--stations", nargs="+", default=None)
    parser.add_argument("--phase2_dir", default="src/workflow/results/lgcp_twostage")
    parser.add_argument("--n_mc", type=int, default=500,
                        help="MC samples for PLN CDF (default: 500)")
    parser.add_argument("--min_days", type=int, default=5)

    args = parser.parse_args()

    run_lgcp_gof(
        count_type=args.count_type,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
        phase2_dir=Path(args.phase2_dir),
        min_days=args.min_days,
        n_mc=args.n_mc,
    )
