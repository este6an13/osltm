"""
Full Bayesian LGCP via Laplace Approximation (Phase 3 — Cox Process Diagnostics)

For each (station, day_type), computes the posterior distribution of the latent
log-intensity z = (z₁,…,z_K) given the observed counts, using:
  - GP prior from Phase 2 kernel parameters
  - Poisson likelihood per bin
  - MAP estimate via L-BFGS-B
  - Laplace posterior (Gaussian at the MAP)

Outputs:
  - lgcp_posterior_{s}_{d}_{ct}.png    — posterior mean Λ(t) with 95% credible band
  - lgcp_posterior_params_{ct}.csv     — per (s,d,k): posterior mean and std of z
  - lgcp_predictive_fano_{ct}.csv      — model-implied vs observed Fano factor
  - lgcp_predictive_fano_{ct}.png      — Fano factor comparison plot
"""

import json
import warnings
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.workflow.data_loader import load_data, load_persisted_data

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]

EPS_0 = 0.5  # Continuity correction (consistent with Phase 2)


# ── Helpers ──────────────────────────────────────────────────────────────────


def sort_time_columns(time_cols: list[str]) -> list[str]:
    return sorted(time_cols, key=lambda c: int(c.replace("t_", "")))


def time_column_to_hours(time_col: str) -> float:
    v = int(time_col.replace("t_", ""))
    return v // 100 + (v % 100) / 60.0


# ── GP Kernels (same as Phase 2) ────────────────────────────────────────────


def kernel_se(t: np.ndarray, sigma2: float, ell: float, eta2: float) -> np.ndarray:
    diff = t[:, None] - t[None, :]
    K = sigma2 * np.exp(-0.5 * (diff / ell) ** 2)
    K += eta2 * np.eye(len(t))
    return K


def kernel_matern32(t: np.ndarray, sigma2: float, ell: float, eta2: float) -> np.ndarray:
    r = np.abs(t[:, None] - t[None, :])
    sqrt3_r_l = np.sqrt(3.0) * r / ell
    K = sigma2 * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
    K += eta2 * np.eye(len(t))
    return K


# ── Laplace Approximation ───────────────────────────────────────────────────


def log_posterior(
    z: np.ndarray,
    n: np.ndarray,
    mu: np.ndarray,
    C_inv: np.ndarray,
    delta_t: float,
) -> float:
    """
    Unnormalised log-posterior: log p(z | n, θ).

    log p(z | n) ∝  Σ_k [n_k · z_k − exp(z_k) · Δt]
                    − ½ (z − μ)ᵀ C⁻¹ (z − μ)
    """
    residual = z - mu
    log_lik = np.sum(n * z - np.exp(z) * delta_t)
    log_prior = -0.5 * residual @ C_inv @ residual
    return log_lik + log_prior


def neg_log_posterior(
    z: np.ndarray,
    n: np.ndarray,
    mu: np.ndarray,
    C_inv: np.ndarray,
    delta_t: float,
) -> float:
    return -log_posterior(z, n, mu, C_inv, delta_t)


def grad_neg_log_posterior(
    z: np.ndarray,
    n: np.ndarray,
    mu: np.ndarray,
    C_inv: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """
    Gradient of −log p(z | n):
      ∂/∂z_k = −n_k + exp(z_k)·Δt + [C⁻¹(z − μ)]_k
    """
    return -n + np.exp(z) * delta_t + C_inv @ (z - mu)


def hessian_neg_log_posterior(
    z: np.ndarray,
    delta_t: float,
    C_inv: np.ndarray,
) -> np.ndarray:
    """
    Hessian of −log p(z | n):
      H_{kl} = diag(exp(z_k)·Δt) + C⁻¹
    """
    K = len(z)
    W = np.diag(np.exp(z) * delta_t)  # Poisson curvature
    return W + C_inv


def find_map(
    n: np.ndarray,
    mu: np.ndarray,
    C_inv: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, bool]:
    """
    Find MAP estimate ẑ = argmax log p(z | n, θ) via L-BFGS-B.

    Initialisation: z₀ = log(n + ε₀) (a sensible starting point).
    """
    z0 = np.log(n + EPS_0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            neg_log_posterior,
            z0,
            args=(n, mu, C_inv, delta_t),
            jac=grad_neg_log_posterior,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

    return result.x, result.success


def laplace_posterior(
    z_map: np.ndarray,
    delta_t: float,
    C_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Laplace approximation: posterior ≈ N(ẑ, Σ) where Σ = H⁻¹.

    Returns:
        z_map: MAP estimate (posterior mean)
        z_std: marginal posterior std per bin (sqrt of diag(H⁻¹))
    """
    H = hessian_neg_log_posterior(z_map, delta_t, C_inv)
    try:
        L = np.linalg.cholesky(H)
        # Σ = H⁻¹, diag(Σ) via solving
        I = np.eye(len(z_map))
        Sigma = np.linalg.solve(H, I)
        z_std = np.sqrt(np.maximum(np.diag(Sigma), 0.0))
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        Sigma = np.linalg.pinv(H)
        z_std = np.sqrt(np.maximum(np.diag(Sigma), 0.0))

    return z_map, z_std


def prior_predictive_fano(
    mu: np.ndarray,
    C: np.ndarray,
    delta_t: float,
    n_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute model-implied Fano factor via Monte Carlo from the GP prior.

    Simulates day-to-day variability by sampling from the fitted LGCP:
        z_j ~ N(μ, C_θ)              ← day-level GP draw
        N_{j,k} ~ Poisson(exp(z_{j,k}) · Δt)

    The Fano factor is Var(N_k) / E(N_k) across simulated days,
    which should match the observed Fano if the LGCP is correct.

    Returns:
        fano: Fano factor per bin
        means: mean counts per bin from the simulation
    """
    rng = np.random.default_rng(seed)
    K = len(mu)

    # Sample latent z from the GP prior (day-level variability)
    try:
        L = np.linalg.cholesky(C + 1e-8 * np.eye(K))
        z_samples = mu[None, :] + (L @ rng.standard_normal((K, n_samples))).T
    except np.linalg.LinAlgError:
        # Fallback: diagonal sampling
        z_std = np.sqrt(np.maximum(np.diag(C), 0.0))
        z_samples = rng.normal(mu[None, :], z_std[None, :], size=(n_samples, K))

    lambda_samples = np.exp(z_samples) * delta_t  # (n_samples, K)

    # Sample counts from Poisson
    n_counts = rng.poisson(lambda_samples)  # (n_samples, K)

    # Compute Fano per bin across simulated days
    means = n_counts.mean(axis=0)
    variances = n_counts.var(axis=0, ddof=1)

    fano = np.where(means > 0, variances / means, np.nan)

    return fano, means


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_posterior_intensity(
    time_hours: np.ndarray,
    counts_all: np.ndarray,
    z_map: np.ndarray,
    z_std: np.ndarray,
    delta_t: float,
    station_code: str,
    day_type: str,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot posterior mean intensity with 95% credible band,
    overlaid with observed individual-day profiles and mean profile.
    """
    J, K = counts_all.shape

    # Posterior intensity (counts per bin, not per minute)
    lambda_map = np.exp(z_map) * delta_t
    lambda_lo = np.exp(z_map - 1.96 * z_std) * delta_t
    lambda_hi = np.exp(z_map + 1.96 * z_std) * delta_t

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual days (light grey)
    for j in range(min(J, 30)):  # cap at 30 for readability
        ax.plot(time_hours, counts_all[j, :], color="grey", alpha=0.1, linewidth=0.5)

    # Observed mean
    obs_mean = counts_all.mean(axis=0)
    ax.plot(time_hours, obs_mean, "ko-", markersize=2, linewidth=1.5,
            label="Observed mean", zorder=3)

    # Posterior credible band
    ax.fill_between(time_hours, lambda_lo, lambda_hi,
                     alpha=0.3, color="steelblue", label="95% credible band")
    ax.plot(time_hours, lambda_map, "-", color="steelblue", linewidth=2,
            label="Posterior mean Λ(t)", zorder=4)

    dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
    ax.set_xlabel("Time of Day (hours)", fontsize=12)
    ax.set_ylabel("Counts per 15-min bin", fontsize=12)
    ax.set_title(
        f"LGCP Posterior — {station_code} {dt_label} ({count_type.capitalize()})\n"
        f"J={J} days",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_hours[0] - 0.3, time_hours[-1] + 0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"lgcp_posterior_{station_code}_{day_type}_{count_type}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  💾 {fname.name}")
    plt.close()


def plot_predictive_fano(
    fit_results: list[dict],
    time_cols: list[str],
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot model-implied Fano factor vs observed, per day-type
    (median across stations with IQR).
    """
    time_hours = np.array([time_column_to_hours(tc) for tc in time_cols])

    n_daytypes = len(DATE_TYPE_ORDER)
    fig, axes = plt.subplots(n_daytypes, 1, figsize=(12, 4 * n_daytypes), sharex=True)
    if n_daytypes == 1:
        axes = [axes]

    for idx, day_type in enumerate(DATE_TYPE_ORDER):
        ax = axes[idx]
        dt_results = [r for r in fit_results if r["day_type"] == day_type]

        if not dt_results:
            dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
            ax.text(0.5, 0.5, f"No data for {dt_label}",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{dt_label}", fontsize=14, fontweight="bold")
            continue

        # Collect per-station arrays
        obs_fanos = np.array([r["observed_fano"] for r in dt_results])  # (S, K)
        pred_fanos = np.array([r["predictive_fano"] for r in dt_results])  # (S, K)

        # Median across stations
        obs_med = np.nanmedian(obs_fanos, axis=0)
        pred_med = np.nanmedian(pred_fanos, axis=0)
        pred_q25 = np.nanpercentile(pred_fanos, 25, axis=0)
        pred_q75 = np.nanpercentile(pred_fanos, 75, axis=0)

        valid = ~(np.isnan(obs_med) | np.isnan(pred_med))
        if not np.any(valid):
            continue

        ax.plot(time_hours[valid], obs_med[valid], "ko-", markersize=3,
                linewidth=1.5, label="Observed Fano")
        ax.fill_between(time_hours[valid], pred_q25[valid], pred_q75[valid],
                         alpha=0.3, color="steelblue", label="Model IQR")
        ax.plot(time_hours[valid], pred_med[valid], "-", color="steelblue",
                linewidth=2, label="Model median Fano")
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                   label="Poisson (Fano=1)")

        dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
        ax.set_ylabel("Fano Factor", fontsize=11)
        ax.set_title(f"{dt_label} ({count_type.capitalize()})",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_hours[0] - 0.5, time_hours[-1] + 0.5)

    axes[-1].set_xlabel("Time of Day (hours)", fontsize=12)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"lgcp_predictive_fano_{count_type}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"💾 Saved predictive Fano plot to {fname}")
    plt.close()


# ── Main computation ─────────────────────────────────────────────────────────


def run_lgcp_bayesian(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    min_days: int = 10,
    phase2_dir: Optional[Path] = None,
) -> dict:
    """
    Run full Bayesian LGCP via Laplace approximation.

    Uses kernel parameters from Phase 2 as the GP prior.
    """
    # ── Load parameters & data ────────────────────────────────────────────
    if params_path is None:
        params_path = Path("src/workflow/params.json")
    with open(params_path) as f:
        params = json.load(f)

    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)
    # Phase 2 GP was fitted to log(counts per bin), so exp(z) = expected
    # count per bin, NOT a rate per minute.  delta_t = 1.0 keeps the
    # Poisson likelihood in count space: N_k ~ Poisson(exp(z_k) · 1).
    delta_t = 1.0

    persistence_dir = Path(step4_params.get("persistence_dir", "src/workflow/data"))
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)
    if not sampled_dates:
        raise ValueError("No sampled dates found. Run workflow steps 1-4 first.")

    if sampled_stations:
        available = [s["code"] for s in sampled_stations]
        station_codes = (
            [sc for sc in station_codes if sc in available]
            if station_codes
            else available
        )
    elif station_codes is None:
        raise ValueError("No station codes provided and none found in data.")

    # ── Load Phase 2 kernel params ────────────────────────────────────────
    if phase2_dir is None:
        phase2_dir = Path("src/workflow/results/lgcp_twostage")
    phase2_dir = Path(phase2_dir)

    kernel_params_file = phase2_dir / f"lgcp_kernel_params_{count_type}.csv"
    if not kernel_params_file.exists():
        raise FileNotFoundError(
            f"Phase 2 results not found at {kernel_params_file}. "
            "Run lgcp_twostage first."
        )

    kernel_params_df = pd.read_csv(kernel_params_file)
    # Ensure station_code is zero-padded string (CSV may store as int)
    kernel_params_df["station_code"] = (
        kernel_params_df["station_code"].astype(str).str.zfill(5)
    )
    # Keep only selected kernels
    selected_kernels = kernel_params_df[kernel_params_df["is_selected"] == True].copy()  # noqa: E712
    print(f"📂 Loaded {len(selected_kernels)} kernel fits from Phase 2")

    # ── Load count data ───────────────────────────────────────────────────
    print(f"📊 Loading {count_type} data...")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min, time_max=time_max, time_step=time_step,
    )
    if count_type not in data:
        raise ValueError(f"No {count_type} data available")
    df = data[count_type]
    if df.empty:
        raise ValueError(f"Empty DataFrame for {count_type}")

    time_cols = sort_time_columns([c for c in df.columns if c.startswith("t_")])
    K = len(time_cols)
    time_hours = np.array([time_column_to_hours(tc) for tc in time_cols])

    if output_dir is None:
        output_dir = Path("src/workflow/results/lgcp_bayesian")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Loop over (station, day_type) ─────────────────────────────────────
    posterior_rows: list[dict] = []
    fano_rows: list[dict] = []
    fit_results: list[dict] = []

    for _, krow in selected_kernels.iterrows():
        station_code = krow["station_code"]
        day_type = krow["day_type"]
        kernel_name = krow["kernel"]
        sigma2 = krow["sigma2"]
        ell = krow["ell_hours"]
        eta2 = krow["eta2"]
        dt_label = DATE_TYPE_LABELS.get(day_type, day_type)

        # Get count data for this (s, d)
        sdf = df[(df["station_code"] == station_code) & (df["date_type"] == day_type)]
        J = len(sdf)
        if J < min_days:
            continue

        print(f"\n🔬 {station_code} / {dt_label}  (J={J}, kernel={kernel_name})")

        counts_all = sdf[time_cols].fillna(0).values.astype(float)  # (J, K)
        mean_counts = counts_all.mean(axis=0)  # (K,)

        # ── Build GP prior ────────────────────────────────────────────────
        kernel_fn = kernel_se if kernel_name == "SE" else kernel_matern32
        C = kernel_fn(time_hours, sigma2, ell, eta2)

        # Regularise for numerical stability
        C += 1e-6 * np.eye(K)
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            C_inv = np.linalg.pinv(C)

        # Mean log-intensity (prior mean)
        mu = np.log(mean_counts + EPS_0)

        # ── MAP on the mean profile ───────────────────────────────────────
        # We find the MAP for the "population" intensity given the mean counts
        # This is more stable than per-day MAP and gives a population-level result
        print("   Finding MAP estimate...")
        z_map, converged = find_map(mean_counts, mu, C_inv, delta_t)
        print(f"   MAP converged: {converged}")

        # ── Laplace posterior ─────────────────────────────────────────────
        print("   Computing Laplace posterior...")
        z_post, z_std = laplace_posterior(z_map, delta_t, C_inv)

        # Posterior intensity
        lambda_map = np.exp(z_post) * delta_t
        lambda_lo = np.exp(z_post - 1.96 * z_std) * delta_t
        lambda_hi = np.exp(z_post + 1.96 * z_std) * delta_t

        print(f"   Posterior Λ: mean={lambda_map.mean():.2f}, "
              f"range=[{lambda_map.min():.2f}, {lambda_map.max():.2f}]")

        # ── Plot posterior intensity ──────────────────────────────────────
        plot_posterior_intensity(
            time_hours, counts_all, z_post, z_std, delta_t,
            station_code, day_type,
            output_dir=output_dir, count_type=count_type,
        )

        # ── Prior predictive Fano factor ───────────────────────────────
        # Build smooth kernel WITHOUT nugget: the nugget η² in Phase 2
        # absorbed Poisson noise from the log-transform (1/Λ term from
        # the delta method). The Poisson variability is added separately
        # by the Poisson sampling step in prior_predictive_fano.
        C_smooth = kernel_fn(time_hours, sigma2, ell, 0.0)
        C_smooth += 1e-6 * np.eye(K)  # minimal regularisation only

        print("   Computing prior-predictive Fano factor...")
        pred_fano, pred_means = prior_predictive_fano(mu, C_smooth, delta_t)

        # Observed Fano
        obs_var = counts_all.var(axis=0, ddof=1)
        obs_mean = counts_all.mean(axis=0)
        obs_fano = np.where(obs_mean > 0, obs_var / obs_mean, np.nan)

        fit_results.append({
            "station_code": station_code,
            "day_type": day_type,
            "observed_fano": obs_fano,
            "predictive_fano": pred_fano,
        })

        # ── Store results ─────────────────────────────────────────────────
        for k in range(K):
            posterior_rows.append({
                "station_code": station_code,
                "day_type": day_type,
                "time_bin": time_cols[k],
                "z_posterior_mean": z_post[k],
                "z_posterior_std": z_std[k],
                "lambda_posterior_mean": lambda_map[k],
                "lambda_95_lo": lambda_lo[k],
                "lambda_95_hi": lambda_hi[k],
                "observed_mean": obs_mean[k],
            })

            fano_rows.append({
                "station_code": station_code,
                "day_type": day_type,
                "time_bin": time_cols[k],
                "observed_fano": obs_fano[k],
                "predictive_fano": pred_fano[k],
            })

    # ── Save CSVs ─────────────────────────────────────────────────────────
    if posterior_rows:
        posterior_df = pd.DataFrame(posterior_rows)
    else:
        posterior_df = pd.DataFrame(
            columns=["station_code", "day_type", "time_bin",
                     "z_posterior_mean", "z_posterior_std",
                     "lambda_posterior_mean", "lambda_95_lo", "lambda_95_hi",
                     "observed_mean"]
        )
    posterior_csv = output_dir / f"lgcp_posterior_params_{count_type}.csv"
    posterior_df.to_csv(posterior_csv, index=False)
    print(f"\n💾 Saved posterior parameters to {posterior_csv}")

    if fano_rows:
        fano_df = pd.DataFrame(fano_rows)
    else:
        fano_df = pd.DataFrame(
            columns=["station_code", "day_type", "time_bin",
                     "observed_fano", "predictive_fano"]
        )
    fano_csv = output_dir / f"lgcp_predictive_fano_{count_type}.csv"
    fano_df.to_csv(fano_csv, index=False)
    print(f"💾 Saved predictive Fano factors to {fano_csv}")

    # ── Predictive Fano plot ──────────────────────────────────────────────
    print("📊 Generating predictive Fano comparison plot...")
    plot_predictive_fano(fit_results, time_cols, output_dir=output_dir, count_type=count_type)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n📋 Summary:")
    for day_type in DATE_TYPE_ORDER:
        dt_fano = fano_df[fano_df["day_type"] == day_type]
        if dt_fano.empty:
            continue
        dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
        obs_med = dt_fano["observed_fano"].median()
        pred_med = dt_fano["predictive_fano"].median()
        print(f"   {dt_label}: observed median Fano={obs_med:.2f}, "
              f"model median Fano={pred_med:.2f}")

    print(f"\n✅ Completed Bayesian LGCP analysis for {count_type}")

    return {
        "posterior_df": posterior_df,
        "predictive_fano_df": fano_df,
        "fit_results": fit_results,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Full Bayesian LGCP via Laplace approximation. "
            "Uses Phase 2 kernel parameters as GP prior."
        )
    )
    parser.add_argument(
        "--count_type",
        choices=["checkins", "checkouts"],
        default="checkins",
        help="Type of counts to analyze (default: checkins)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/lgcp_bayesian",
        help="Directory to save results (default: src/workflow/results/lgcp_bayesian)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="src/workflow/params.json",
        help="Path to params.json (default: src/workflow/params.json)",
    )
    parser.add_argument(
        "--stations",
        type=str,
        nargs="+",
        help="Optional list of station codes to analyze (default: all from data)",
    )
    parser.add_argument(
        "--min_days",
        type=int,
        default=10,
        help="Minimum replicate days required per (station, day_type) (default: 10)",
    )
    parser.add_argument(
        "--phase2_dir",
        type=str,
        default="src/workflow/results/lgcp_twostage",
        help="Directory with Phase 2 results (default: src/workflow/results/lgcp_twostage)",
    )

    args = parser.parse_args()

    results = run_lgcp_bayesian(
        count_type=args.count_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        params_path=Path(args.params),
        station_codes=args.stations,
        min_days=args.min_days,
        phase2_dir=Path(args.phase2_dir) if args.phase2_dir else None,
    )

    print(f"\n📊 Generated Bayesian LGCP analysis for {args.count_type}")
