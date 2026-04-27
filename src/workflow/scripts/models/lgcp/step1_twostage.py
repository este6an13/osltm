"""
LGCP Two-Stage Estimation (Phase 2 — Cox Process Diagnostics)

For each (station, day_type), fits a Log-Gaussian Cox Process via two-stage
estimation:
  1. Estimate mean log-intensity μ̂(t) from sample mean profiles
  2. Compute log-residuals e_{j,k} = log(N_{j,k} + ε₀) − μ̂(t_k)
  3. Estimate empirical covariance Ĉ(t_k, t_l) across replicate days
  4. Fit parametric GP kernels (SE + Matérn 3/2) via MLE on the residuals
  5. Validate Gaussianity (Shapiro–Wilk per bin, Mahalanobis distances)

Outputs:
  - lgcp_kernel_params_{ct}.csv       — per (s,d): fitted kernel parameters
  - lgcp_summary_{ct}.csv            — aggregated summary across stations
  - lgcp_gaussianity_{ct}.csv        — per (s,d,k): Shapiro–Wilk results
  - lgcp_empirical_cov_{s}_{d}_{ct}.png  — covariance heatmaps
  - lgcp_kernel_fit_{s}_{d}_{ct}.png     — empirical vs fitted kernel
  - lgcp_residual_qq_{s}_{d}_{ct}.png    — Mahalanobis QQ-plot
"""

import json
import warnings
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from src.workflow.data_loader import load_data, load_persisted_data

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]

EPS_0 = 0.5  # Continuity correction for log-transform


# ── Helpers ──────────────────────────────────────────────────────────────────


def sort_time_columns(time_cols: list[str]) -> list[str]:
    """Sort time column names by their numeric time value."""
    return sorted(time_cols, key=lambda c: int(c.replace("t_", "")))


def time_column_to_hours(time_col: str) -> float:
    """Convert time column name (e.g., 't_400') to hours of day."""
    v = int(time_col.replace("t_", ""))
    return v // 100 + (v % 100) / 60.0


# ── GP Kernels ───────────────────────────────────────────────────────────────


def kernel_se(t: np.ndarray, sigma2: float, ell: float, eta2: float) -> np.ndarray:
    """
    Squared Exponential kernel matrix + nugget.

    C(t_i, t_j) = σ² exp(−(t_i − t_j)²/(2ℓ²)) + η² δ_{ij}
    """
    diff = t[:, None] - t[None, :]
    K = sigma2 * np.exp(-0.5 * (diff / ell) ** 2)
    K += eta2 * np.eye(len(t))
    return K


def kernel_matern32(t: np.ndarray, sigma2: float, ell: float, eta2: float) -> np.ndarray:
    """
    Matérn 3/2 kernel matrix + nugget.

    C(t_i, t_j) = σ² (1 + √3|t_i−t_j|/ℓ) exp(−√3|t_i−t_j|/ℓ) + η² δ_{ij}
    """
    r = np.abs(t[:, None] - t[None, :])
    sqrt3_r_l = np.sqrt(3.0) * r / ell
    K = sigma2 * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
    K += eta2 * np.eye(len(t))
    return K


# ── GP Log-Likelihood ────────────────────────────────────────────────────────


def gp_negloglik(
    log_params: np.ndarray,
    t: np.ndarray,
    residuals: np.ndarray,
    kernel_fn,
) -> float:
    """
    Negative log-likelihood of J i.i.d. draws from N(0, C_θ).

    ℓ(θ) = −(J/2) log|C_θ| − (1/2) Σ_j e_j' C_θ⁻¹ e_j − (JK/2) log(2π)

    Args:
        log_params: [log(σ²), log(ℓ), log(η²)]
        t: time points (K,)
        residuals: (J, K) matrix of residual vectors
        kernel_fn: kernel function (t, σ², ℓ, η²) → (K, K)
    """
    sigma2, ell, eta2 = np.exp(log_params)
    J, K = residuals.shape

    try:
        C = kernel_fn(t, sigma2, ell, eta2)
        # Use Cholesky for stable computation
        L = np.linalg.cholesky(C)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

        # Solve C⁻¹ e_j via Cholesky
        alpha = np.linalg.solve(L, residuals.T)  # (K, J)
        quad_form = np.sum(alpha ** 2)

        nll = 0.5 * J * log_det + 0.5 * quad_form + 0.5 * J * K * np.log(2 * np.pi)
        return nll
    except np.linalg.LinAlgError:
        return 1e30


def fit_gp_kernel(
    t: np.ndarray,
    residuals: np.ndarray,
    kernel_fn,
    kernel_name: str,
) -> dict:
    """
    Fit a GP kernel to the residual matrix via MLE.

    Args:
        t: time points in hours (K,)
        residuals: (J, K) matrix
        kernel_fn: kernel function
        kernel_name: "SE" or "Matern32"

    Returns:
        dict with sigma2, ell, eta2, loglik, aic, bic, converged
    """
    J, K = residuals.shape

    # Empirical initial estimates
    emp_var = np.mean(np.var(residuals, axis=0))
    emp_var = max(emp_var, 1e-4)

    # Initial: σ² = empirical variance, ℓ = 1 hour, η² = 10% of σ²
    x0 = np.log([emp_var, 1.0, emp_var * 0.1])

    # Bounds in log-space
    bounds = [
        (np.log(1e-6), np.log(1e4)),   # log(σ²)
        (np.log(0.1), np.log(20.0)),    # log(ℓ) — 6 min to 20 hours
        (np.log(1e-8), np.log(1e2)),    # log(η²)
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            gp_negloglik,
            x0,
            args=(t, residuals, kernel_fn),
            method="L-BFGS-B",
            bounds=bounds,
        )

    sigma2, ell, eta2 = np.exp(result.x)
    nll = result.fun
    loglik = -nll
    n_params = 3
    n_obs = J * K

    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n_obs) - 2 * loglik

    return {
        "kernel": kernel_name,
        "sigma2": sigma2,
        "ell": ell,
        "eta2": eta2,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "converged": result.success,
    }


# ── Gaussianity Checks ──────────────────────────────────────────────────────


def shapiro_wilk_per_bin(
    residuals: np.ndarray,
    time_cols: list[str],
) -> list[dict]:
    """
    Shapiro–Wilk test per bin k on residual column {e_{j,k}}_j.
    """
    J, K = residuals.shape
    results = []
    for k in range(K):
        col = residuals[:, k]
        # Need at least 3 observations for Shapiro-Wilk
        if J >= 3 and np.std(col) > 1e-10:
            w_stat, p_value = stats.shapiro(col)
        else:
            w_stat, p_value = np.nan, np.nan
        results.append({
            "time_bin": time_cols[k],
            "shapiro_w": w_stat,
            "shapiro_p": p_value,
            "n_days": J,
        })
    return results


def mahalanobis_distances(
    residuals: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute Mahalanobis distance D²_j = e_j' Ĉ⁻¹ e_j for each day j.

    Under Gaussianity, D²_j ~ χ²_K.
    """
    J, K = residuals.shape
    try:
        # Regularise covariance slightly for numerical stability
        C_reg = cov_matrix + 1e-6 * np.eye(K)
        L = np.linalg.cholesky(C_reg)
        alpha = np.linalg.solve(L, residuals.T)  # (K, J)
        D2 = np.sum(alpha ** 2, axis=0)  # (J,)
        return D2
    except np.linalg.LinAlgError:
        return np.full(J, np.nan)


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_empirical_covariance(
    emp_cov: np.ndarray,
    time_hours: np.ndarray,
    station_code: str,
    day_type: str,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """Heatmap of the K×K empirical covariance matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        emp_cov,
        cmap="RdBu_r",
        aspect="auto",
        origin="lower",
        vmin=-np.max(np.abs(emp_cov)),
        vmax=np.max(np.abs(emp_cov)),
    )
    plt.colorbar(im, ax=ax, label="Covariance")

    # Tick labels
    n_ticks = min(10, len(time_hours))
    tick_idx = np.linspace(0, len(time_hours) - 1, n_ticks, dtype=int)
    tick_labels = [f"{time_hours[i]:.1f}h" for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Time of Day", fontsize=12)
    dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
    ax.set_title(
        f"Empirical Covariance — {station_code} {dt_label}\n({count_type.capitalize()})",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"lgcp_empirical_cov_{station_code}_{day_type}_{count_type}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  💾 {fname.name}")
    plt.close()


def plot_kernel_fit(
    emp_cov: np.ndarray,
    time_hours: np.ndarray,
    fits: list[dict],
    station_code: str,
    day_type: str,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Compare the diagonal (variance) and a sample off-diagonal slice of the
    empirical covariance with the fitted parametric kernels.
    """
    K = len(time_hours)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Diagonal (variance per bin) ---
    ax = axes[0]
    ax.plot(time_hours, np.diag(emp_cov), "ko-", markersize=3, label="Empirical", linewidth=1.5)
    for fit in fits:
        kernel_fn = kernel_se if fit["kernel"] == "SE" else kernel_matern32
        C_fit = kernel_fn(time_hours, fit["sigma2"], fit["ell"], fit["eta2"])
        ax.plot(
            time_hours, np.diag(C_fit), "--", linewidth=1.5,
            label=f'{fit["kernel"]} (AIC={fit["aic"]:.1f})',
        )
    ax.set_xlabel("Time of Day (hours)", fontsize=11)
    ax.set_ylabel("Variance", fontsize=11)
    ax.set_title("Diagonal: C(t, t)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Off-diagonal slice at the mid-point ---
    mid = K // 2
    ax = axes[1]
    ax.plot(time_hours, emp_cov[mid, :], "ko-", markersize=3, label="Empirical", linewidth=1.5)
    for fit in fits:
        kernel_fn = kernel_se if fit["kernel"] == "SE" else kernel_matern32
        C_fit = kernel_fn(time_hours, fit["sigma2"], fit["ell"], fit["eta2"])
        ax.plot(
            time_hours, C_fit[mid, :], "--", linewidth=1.5,
            label=f'{fit["kernel"]}',
        )
    mid_h = time_hours[mid]
    ax.set_xlabel("Time of Day (hours)", fontsize=11)
    ax.set_ylabel(f"Cov(t, {mid_h:.1f}h)", fontsize=11)
    ax.set_title(f"Slice at t={mid_h:.1f}h", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
    fig.suptitle(
        f"Kernel Fit — {station_code} {dt_label} ({count_type.capitalize()})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"lgcp_kernel_fit_{station_code}_{day_type}_{count_type}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  💾 {fname.name}")
    plt.close()


def plot_mahalanobis_qq(
    D2: np.ndarray,
    K: int,
    station_code: str,
    day_type: str,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """QQ-plot of Mahalanobis D² against χ²_K."""
    D2_valid = D2[np.isfinite(D2)]
    if len(D2_valid) < 3:
        return

    J = len(D2_valid)
    theoretical = stats.chi2.ppf(
        np.linspace(1 / (J + 1), J / (J + 1), J), df=K
    )
    sample = np.sort(D2_valid)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(theoretical, sample, alpha=0.6, s=25)
    max_val = max(theoretical.max(), sample.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="y = x")

    # KS test against χ²_K
    ks_stat, ks_p = stats.kstest(D2_valid, lambda x: stats.chi2.cdf(x, df=K))
    ax.text(
        0.05, 0.95,
        f"KS stat: {ks_stat:.4f}\np-value: {ks_p:.4f}\nK={K}, J={J}",
        transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
    ax.set_xlabel(f"χ²({K}) Theoretical Quantiles", fontsize=12)
    ax.set_ylabel("Mahalanobis D² Sample Quantiles", fontsize=12)
    ax.set_title(
        f"Mahalanobis QQ — {station_code} {dt_label}\n({count_type.capitalize()})",
        fontsize=13, fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"lgcp_residual_qq_{station_code}_{day_type}_{count_type}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  💾 {fname.name}")
    plt.close()


# ── Main computation ─────────────────────────────────────────────────────────


def run_lgcp_twostage(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    min_days: int = 10,
) -> dict:
    """
    Run LGCP two-stage estimation for all (station, day_type) pairs.

    Args:
        count_type: "checkins" or "checkouts"
        output_dir: Where to save results
        params_path: Path to params.json
        station_codes: Optional station filter
        min_days: Minimum replicate days required per (s, d)

    Returns:
        Dict with kernel_params_df, gaussianity_df, summary_df.
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

    persistence_dir = params_path.parent
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
        output_dir = Path("src/workflow/results/lgcp_twostage")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Loop over (station, day_type) ─────────────────────────────────────
    kernel_rows: list[dict] = []
    gauss_rows: list[dict] = []

    stations = sorted(df["station_code"].unique())
    print(f"   Stations: {len(stations)}, bins: {K}")

    for station_code in stations:
        sdf = df[df["station_code"] == station_code]

        for day_type in sorted(sdf["date_type"].unique()):
            dtdf = sdf[sdf["date_type"] == day_type]
            J = len(dtdf)
            if J < min_days:
                continue

            dt_label = DATE_TYPE_LABELS.get(day_type, day_type)
            print(f"\n🔬 {station_code} / {dt_label}  (J={J})")

            counts = dtdf[time_cols].fillna(0).values.astype(float)  # (J, K)

            # ── Step 1: mean log-intensity ────────────────────────────────
            mu = counts.mean(axis=0)  # (K,)
            m_hat = np.log(mu + EPS_0)  # (K,)

            # ── Step 2: log-residuals ─────────────────────────────────────
            residuals = np.log(counts + EPS_0) - m_hat[None, :]  # (J, K)

            # ── Step 3: empirical covariance ──────────────────────────────
            emp_cov = np.cov(residuals, rowvar=False, ddof=1)  # (K, K)

            plot_empirical_covariance(
                emp_cov, time_hours, station_code, day_type,
                output_dir=output_dir, count_type=count_type,
            )

            # ── Step 4: kernel fitting ────────────────────────────────────
            print("   Fitting SE kernel...")
            fit_se = fit_gp_kernel(time_hours, residuals, kernel_se, "SE")
            print(f"   σ²={fit_se['sigma2']:.4f}, ℓ={fit_se['ell']:.2f}h, "
                  f"η²={fit_se['eta2']:.6f}, AIC={fit_se['aic']:.1f}")

            print("   Fitting Matérn 3/2 kernel...")
            fit_m32 = fit_gp_kernel(time_hours, residuals, kernel_matern32, "Matern32")
            print(f"   σ²={fit_m32['sigma2']:.4f}, ℓ={fit_m32['ell']:.2f}h, "
                  f"η²={fit_m32['eta2']:.6f}, AIC={fit_m32['aic']:.1f}")

            # Select best kernel by AIC
            best = fit_se if fit_se["aic"] <= fit_m32["aic"] else fit_m32
            print(f"   ✅ Selected: {best['kernel']}")

            # Store results for both kernels
            for fit in (fit_se, fit_m32):
                kernel_rows.append({
                    "station_code": station_code,
                    "day_type": day_type,
                    "n_days": J,
                    "kernel": fit["kernel"],
                    "sigma2": fit["sigma2"],
                    "ell_hours": fit["ell"],
                    "eta2": fit["eta2"],
                    "loglik": fit["loglik"],
                    "aic": fit["aic"],
                    "bic": fit["bic"],
                    "converged": fit["converged"],
                    "is_selected": fit["kernel"] == best["kernel"],
                })

            plot_kernel_fit(
                emp_cov, time_hours, [fit_se, fit_m32],
                station_code, day_type,
                output_dir=output_dir, count_type=count_type,
            )

            # ── Step 5: Gaussianity validation ────────────────────────────
            # 5a. Shapiro–Wilk per bin
            sw_results = shapiro_wilk_per_bin(residuals, time_cols)
            for sw in sw_results:
                gauss_rows.append({
                    "station_code": station_code,
                    "day_type": day_type,
                    **sw,
                })

            n_reject = sum(
                1 for sw in sw_results
                if not np.isnan(sw["shapiro_p"]) and sw["shapiro_p"] < 0.05
            )
            n_valid = sum(1 for sw in sw_results if not np.isnan(sw["shapiro_p"]))
            print(f"   Shapiro–Wilk: {n_reject}/{n_valid} bins reject Normality (α=0.05)")

            # 5b. Mahalanobis QQ-plot
            D2 = mahalanobis_distances(residuals, emp_cov)
            plot_mahalanobis_qq(
                D2, K, station_code, day_type,
                output_dir=output_dir, count_type=count_type,
            )

    # ── Save CSVs ─────────────────────────────────────────────────────────
    kernel_df = pd.DataFrame(kernel_rows)
    kernel_csv = output_dir / f"lgcp_kernel_params_{count_type}.csv"
    kernel_df.to_csv(kernel_csv, index=False)
    print(f"\n💾 Saved kernel parameters to {kernel_csv}")

    gauss_df = pd.DataFrame(gauss_rows)
    gauss_csv = output_dir / f"lgcp_gaussianity_{count_type}.csv"
    gauss_df.to_csv(gauss_csv, index=False)
    print(f"💾 Saved Gaussianity results to {gauss_csv}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_rows = []
    selected = kernel_df[kernel_df["is_selected"] == True]  # noqa: E712

    for day_type in DATE_TYPE_ORDER:
        dt_sel = selected[selected["day_type"] == day_type]
        dt_gauss = gauss_df[gauss_df["day_type"] == day_type]
        if dt_sel.empty:
            continue

        valid_sw = dt_gauss[dt_gauss["shapiro_p"].notna()]
        n_reject = (valid_sw["shapiro_p"] < 0.05).sum() if not valid_sw.empty else 0
        n_total_sw = len(valid_sw)

        summary_rows.append({
            "day_type": day_type,
            "n_station_daytype_pairs": len(dt_sel),
            "pct_se_selected": (dt_sel["kernel"] == "SE").mean() * 100,
            "median_sigma2": dt_sel["sigma2"].median(),
            "median_ell_hours": dt_sel["ell_hours"].median(),
            "median_eta2": dt_sel["eta2"].median(),
            "pct_bins_reject_normality": n_reject / n_total_sw * 100 if n_total_sw > 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"lgcp_summary_{count_type}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"💾 Saved summary to {summary_csv}")

    # Print summary
    print("\n📋 Summary per day-type (selected kernel):")
    for _, row in summary_df.iterrows():
        dt_label = DATE_TYPE_LABELS.get(row["day_type"], row["day_type"])
        print(
            f"   {dt_label}: SE chosen {row['pct_se_selected']:.0f}%, "
            f"median σ²={row['median_sigma2']:.4f}, "
            f"median ℓ={row['median_ell_hours']:.2f}h, "
            f"normality rejected in {row['pct_bins_reject_normality']:.1f}% of bins"
        )

    print(f"\n✅ Completed LGCP two-stage estimation for {count_type}")

    return {
        "kernel_params_df": kernel_df,
        "gaussianity_df": gauss_df,
        "summary_df": summary_df,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "LGCP two-stage estimation: fit GP kernels to log-residual "
            "covariance and validate Gaussianity per (station, day_type)."
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
        default="src/workflow/results/lgcp_twostage",
        help="Directory to save results (default: src/workflow/results/lgcp_twostage)",
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

    args = parser.parse_args()

    results = run_lgcp_twostage(
        count_type=args.count_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        params_path=Path(args.params),
        station_codes=args.stations,
        min_days=args.min_days,
    )

    print(f"\n📊 Generated LGCP two-stage estimation for {args.count_type}")
