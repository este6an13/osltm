"""
Negative Binomial per-bin fit (Phase 1 — Cox Process Diagnostics)

For each (station, day_type, time_bin) triplet, fits both a Poisson and a
Negative Binomial model to the across-days count vector {N_{s,d,j,k}}, j=1..J.

Reports:
  - NegBin dispersion parameter r̂, probability p̂
  - Mean, variance, Fano factor
  - Log-likelihoods
  - AIC comparison (ΔAIC = AIC_Poisson − AIC_NegBin; positive means NegBin wins)

Outputs:
  - negbin_fit_{count_type}.csv          — per (station, day_type, time_bin) row
  - negbin_summary_{count_type}.csv      — aggregated summary
  - negbin_dispersion_{count_type}.png   — 1/r̂ vs time-of-day per day-type
  - negbin_aic_comparison_{count_type}.png — ΔAIC vs time-of-day per day-type
"""

import json
import warnings
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from src.workflow.data_loader import load_data, load_persisted_data

DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}

DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]


def sort_time_columns(time_cols: list[str]) -> list[str]:
    """Sort time column names by their numeric time value."""

    def get_time_value(col: str) -> int:
        time_str = col.replace("t_", "")
        return int(time_str)

    return sorted(time_cols, key=get_time_value)


def time_column_to_hours(time_col: str) -> float:
    """Convert time column name (e.g., 't_400') to hours of day."""
    time_str = time_col.replace("t_", "")
    time_int = int(time_str)
    hour = time_int // 100
    minute = time_int % 100
    return hour + minute / 60.0


# ── Poisson MLE ──────────────────────────────────────────────────────────────


def poisson_loglik(counts: np.ndarray) -> float:
    """
    Poisson log-likelihood at the MLE λ̂ = mean(counts).

    ℓ(λ̂) = Σ_j [ n_j log(λ̂) − λ̂ − log(n_j!) ]
    """
    lam = np.mean(counts)
    if lam <= 0:
        return -np.inf
    return np.sum(counts * np.log(lam) - lam - gammaln(counts + 1))


def poisson_aic(counts: np.ndarray) -> float:
    """AIC for Poisson model (1 parameter: λ)."""
    return 2 * 1 - 2 * poisson_loglik(counts)


# ── Negative Binomial MLE ────────────────────────────────────────────────────


def negbin_loglik_r(r: float, counts: np.ndarray) -> float:
    """
    NegBin log-likelihood as a function of r, with p profiled out.

    Parameterisation: P(N=n) = C(n+r-1, n) p^n (1-p)^r
    where p = μ/(μ+r) and μ = mean(counts).

    ℓ(r) = Σ_j [ log Γ(n_j + r) − log Γ(r) − log(n_j!)
                 + n_j log(p̂) + r log(1 − p̂) ]
    """
    n = counts
    mu = np.mean(n)
    if mu <= 0 or r <= 0:
        return -np.inf
    p = mu / (mu + r)
    ll = np.sum(
        gammaln(n + r)
        - gammaln(r)
        - gammaln(n + 1)
        + n * np.log(p + 1e-300)
        + r * np.log(1 - p + 1e-300)
    )
    return ll


def fit_negbin(counts: np.ndarray) -> dict:
    """
    Fit Negative Binomial to a count vector via profile-likelihood for r.

    Returns dict with keys: r, p, loglik, converged.
    """
    mu = np.mean(counts)
    var = np.var(counts, ddof=1)

    if mu <= 0 or var <= mu:
        # Variance ≤ mean: NegBin not needed / r → ∞ (Poisson limit)
        return {"r": np.inf, "p": 1.0, "loglik": poisson_loglik(counts), "converged": False}

    # Method-of-moments initial estimate: r̂_mom = μ² / (σ² − μ)
    r_init = mu ** 2 / (var - mu)

    # Profile-likelihood optimisation (minimise negative LL over r)
    def neg_ll(log_r: float) -> float:
        r = np.exp(log_r)
        ll = negbin_loglik_r(r, counts)
        return -ll if np.isfinite(ll) else 1e30

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize_scalar(
            neg_ll,
            bounds=(np.log(1e-4), np.log(1e6)),
            method="bounded",
        )

    r_hat = np.exp(result.x)
    p_hat = mu / (mu + r_hat)
    ll = negbin_loglik_r(r_hat, counts)

    return {"r": r_hat, "p": p_hat, "loglik": ll, "converged": result.success}


def negbin_aic(counts: np.ndarray, negbin_result: dict) -> float:
    """AIC for NegBin model (2 parameters: r, p)."""
    return 2 * 2 - 2 * negbin_result["loglik"]


# ── Main computation ─────────────────────────────────────────────────────────


def fit_all_bins(
    df: pd.DataFrame,
    min_days: int = 5,
) -> pd.DataFrame:
    """
    Fit Poisson and NegBin per (station, day_type, time_bin).

    Args:
        df: DataFrame from data_loader with time columns t_400..t_2300.
        min_days: Minimum number of replicate days required to attempt a fit.

    Returns:
        DataFrame with one row per (station_code, day_type, time_bin).
    """
    time_cols = sort_time_columns(
        [col for col in df.columns if col.startswith("t_")]
    )

    rows: list[dict] = []

    for station_code in sorted(df["station_code"].unique()):
        station_df = df[df["station_code"] == station_code]

        for day_type in sorted(station_df["date_type"].unique()):
            daytype_df = station_df[station_df["date_type"] == day_type]
            J = len(daytype_df)
            if J < min_days:
                continue

            for time_col in time_cols:
                counts = daytype_df[time_col].fillna(0).values.astype(float)

                mu = np.mean(counts)
                var = np.var(counts, ddof=1)
                fano = var / mu if mu > 0 else np.nan

                # Poisson fit
                ll_pois = poisson_loglik(counts)
                aic_pois = 2 * 1 - 2 * ll_pois

                # NegBin fit
                nb = fit_negbin(counts)
                aic_nb = 2 * 2 - 2 * nb["loglik"]

                delta_aic = aic_pois - aic_nb  # positive ⇒ NegBin preferred

                rows.append(
                    {
                        "station_code": station_code,
                        "day_type": day_type,
                        "time_bin": time_col,
                        "n_days": J,
                        "mean": mu,
                        "variance": var,
                        "fano_factor": fano,
                        "poisson_loglik": ll_pois,
                        "poisson_aic": aic_pois,
                        "negbin_r": nb["r"],
                        "negbin_p": nb["p"],
                        "negbin_loglik": nb["loglik"],
                        "negbin_aic": aic_nb,
                        "negbin_converged": nb["converged"],
                        "delta_aic": delta_aic,
                        "preferred_model": (
                            "NegBin" if delta_aic > 0 else "Poisson"
                        ),
                    }
                )

    return pd.DataFrame(rows)


def compute_summary(fit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per–(day_type) summary from the detailed fit table.
    """
    summary_rows = []

    for day_type in DATE_TYPE_ORDER:
        dt_df = fit_df[fit_df["day_type"] == day_type]
        if dt_df.empty:
            continue

        finite_r = dt_df[np.isfinite(dt_df["negbin_r"])]

        summary_rows.append(
            {
                "day_type": day_type,
                "n_bins_total": len(dt_df),
                "n_bins_negbin_preferred": int((dt_df["preferred_model"] == "NegBin").sum()),
                "pct_negbin_preferred": (dt_df["preferred_model"] == "NegBin").mean() * 100,
                "median_fano": dt_df["fano_factor"].median(),
                "median_delta_aic": dt_df["delta_aic"].median(),
                "median_negbin_r": finite_r["negbin_r"].median() if not finite_r.empty else np.nan,
                "median_inv_r": (1.0 / finite_r["negbin_r"]).median() if not finite_r.empty else np.nan,
            }
        )

    return pd.DataFrame(summary_rows)


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_dispersion(
    fit_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot 1/r̂ (overdispersion index) vs time-of-day per day-type.
    Median across stations with IQR envelope.

    1/r̂ → 0 means Poisson; larger values indicate stronger overdispersion.
    """
    time_cols = sort_time_columns(fit_df["time_bin"].unique().tolist())
    time_hours = np.array([time_column_to_hours(tc) for tc in time_cols])

    n_daytypes = len(DATE_TYPE_ORDER)
    fig, axes = plt.subplots(
        n_daytypes, 1, figsize=(12, 4 * n_daytypes), sharex=True
    )
    if n_daytypes == 1:
        axes = [axes]

    for idx, day_type in enumerate(DATE_TYPE_ORDER):
        ax = axes[idx]
        dt_df = fit_df[fit_df["day_type"] == day_type]

        if dt_df.empty:
            ax.text(
                0.5, 0.5,
                f"No data for {DATE_TYPE_LABELS.get(day_type, day_type)}",
                transform=ax.transAxes, ha="center", va="center",
            )
            ax.set_title(
                f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type.capitalize()})",
                fontsize=14, fontweight="bold",
            )
            continue

        medians, q25s, q75s = [], [], []
        for tc in time_cols:
            bin_df = dt_df[dt_df["time_bin"] == tc]
            inv_r = 1.0 / bin_df["negbin_r"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(inv_r) > 0:
                medians.append(np.median(inv_r))
                q25s.append(np.percentile(inv_r, 25))
                q75s.append(np.percentile(inv_r, 75))
            else:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)

        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        valid = ~np.isnan(medians)

        if np.any(valid):
            ax.fill_between(
                time_hours[valid], q25s[valid], q75s[valid],
                alpha=0.2, color="blue", label="IQR across stations",
            )
            ax.plot(
                time_hours[valid], medians[valid],
                "o-", color="blue", linewidth=2, markersize=4,
                label="Median across stations",
            )

        ax.axhline(y=0.0, color="red", linestyle="--", linewidth=2, label="Poisson (1/r = 0)")
        ax.set_ylabel("1/r̂  (overdispersion index)", fontsize=12)
        ax.set_title(
            f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type.capitalize()})",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_hours[0] - 0.5, time_hours[-1] + 0.5)

    axes[-1].set_xlabel("Time of Day (hours)", fontsize=12)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"negbin_dispersion_{count_type}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"💾 Saved dispersion plot to {fname}")
    plt.show()


def plot_delta_aic(
    fit_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    count_type: str = "checkins",
) -> None:
    """
    Plot ΔAIC = AIC_Poisson − AIC_NegBin vs time-of-day per day-type.
    Positive values indicate NegBin is preferred.
    """
    time_cols = sort_time_columns(fit_df["time_bin"].unique().tolist())
    time_hours = np.array([time_column_to_hours(tc) for tc in time_cols])

    n_daytypes = len(DATE_TYPE_ORDER)
    fig, axes = plt.subplots(
        n_daytypes, 1, figsize=(12, 4 * n_daytypes), sharex=True
    )
    if n_daytypes == 1:
        axes = [axes]

    for idx, day_type in enumerate(DATE_TYPE_ORDER):
        ax = axes[idx]
        dt_df = fit_df[fit_df["day_type"] == day_type]

        if dt_df.empty:
            ax.text(
                0.5, 0.5,
                f"No data for {DATE_TYPE_LABELS.get(day_type, day_type)}",
                transform=ax.transAxes, ha="center", va="center",
            )
            ax.set_title(
                f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type.capitalize()})",
                fontsize=14, fontweight="bold",
            )
            continue

        medians, q25s, q75s = [], [], []
        for tc in time_cols:
            bin_df = dt_df[dt_df["time_bin"] == tc]
            daic = bin_df["delta_aic"].dropna()
            if len(daic) > 0:
                medians.append(np.median(daic))
                q25s.append(np.percentile(daic, 25))
                q75s.append(np.percentile(daic, 75))
            else:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)

        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        valid = ~np.isnan(medians)

        if np.any(valid):
            ax.fill_between(
                time_hours[valid], q25s[valid], q75s[valid],
                alpha=0.2, color="green", label="IQR across stations",
            )
            ax.plot(
                time_hours[valid], medians[valid],
                "o-", color="green", linewidth=2, markersize=4,
                label="Median ΔAIC across stations",
            )

        ax.axhline(y=0.0, color="red", linestyle="--", linewidth=2, label="ΔAIC = 0 (equal)")
        ax.set_ylabel("ΔAIC  (Poisson − NegBin)", fontsize=12)
        ax.set_title(
            f"{DATE_TYPE_LABELS.get(day_type, day_type)} ({count_type.capitalize()})",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_hours[0] - 0.5, time_hours[-1] + 0.5)

    axes[-1].set_xlabel("Time of Day (hours)", fontsize=12)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"negbin_aic_comparison_{count_type}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"💾 Saved ΔAIC plot to {fname}")
    plt.show()


# ── Entry point ──────────────────────────────────────────────────────────────


def run_negbin_fit(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    min_days: int = 5,
) -> dict:
    """
    Run Negative Binomial per-bin fit analysis.

    Args:
        count_type: "checkins" or "checkouts"
        output_dir: Where to save results (default: src/workflow/results/negbin_fit)
        params_path: Path to params.json
        station_codes: Optional station filter
        min_days: Minimum replicate days required per (station, day_type) to attempt fit

    Returns:
        Dict with fit_df and summary_df DataFrames.
    """
    # Load parameters
    if params_path is None:
        params_path = Path("src/workflow/params.json")

    with open(params_path) as f:
        params = json.load(f)

    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)

    # Load sampled dates and stations
    persistence_dir = step4_params.get("persistence_dir", Path("src/workflow/data"))
    persistence_dir = Path(persistence_dir)
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)

    if not sampled_dates:
        raise ValueError("No sampled dates found. Run workflow steps 1-4 first.")

    if sampled_stations:
        available_station_codes = [s["code"] for s in sampled_stations]
        if station_codes is None:
            station_codes = available_station_codes
        else:
            station_codes = [
                sc for sc in station_codes if sc in available_station_codes
            ]
    elif station_codes is None:
        raise ValueError("No station codes provided and none found in data.")

    print(f"📊 Loading {count_type} data...")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
    )

    if count_type not in data:
        raise ValueError(f"No {count_type} data available")

    df = data[count_type]
    if df.empty:
        raise ValueError(f"Empty DataFrame for {count_type}")

    print(f"   Stations: {len(station_codes)}")
    print(f"   Date-station rows: {len(df)}")

    # ── Fit ───────────────────────────────────────────────────────────────
    print(f"🔍 Fitting Poisson & NegBin per (station, day_type, time_bin) [min_days={min_days}]...")
    fit_df = fit_all_bins(df, min_days=min_days)
    print(f"✅ Fitted {len(fit_df)} (station, day_type, bin) triplets")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_df = compute_summary(fit_df)
    print("\n📋 Summary per day-type:")
    for _, row in summary_df.iterrows():
        dt_label = DATE_TYPE_LABELS.get(row["day_type"], row["day_type"])
        print(
            f"   {dt_label}: {row['pct_negbin_preferred']:.1f}% bins prefer NegBin, "
            f"median Fano={row['median_fano']:.2f}, "
            f"median ΔAIC={row['median_delta_aic']:.1f}, "
            f"median r̂={row['median_negbin_r']:.2f}"
        )

    # ── Save CSVs ─────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = Path("src/workflow/results/negbin_fit")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_csv = output_dir / f"negbin_fit_{count_type}.csv"
    fit_df.to_csv(fit_csv, index=False)
    print(f"💾 Saved per-bin fit table to {fit_csv}")

    summary_csv = output_dir / f"negbin_summary_{count_type}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"💾 Saved summary to {summary_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("📊 Generating plots...")
    plot_dispersion(fit_df, output_dir=output_dir, count_type=count_type)
    plot_delta_aic(fit_df, output_dir=output_dir, count_type=count_type)

    print(f"\n✅ Completed NegBin fit analysis for {count_type}")

    return {"fit_df": fit_df, "summary_df": summary_df}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fit Poisson & Negative Binomial models per (station, day_type, time_bin). "
            "Reports dispersion parameters and AIC comparison."
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
        default="src/workflow/results/negbin_fit",
        help="Directory to save results (default: src/workflow/results/negbin_fit)",
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
        default=5,
        help="Minimum replicate days required to fit a bin (default: 5)",
    )

    args = parser.parse_args()

    results = run_negbin_fit(
        count_type=args.count_type,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        params_path=Path(args.params),
        station_codes=args.stations,
        min_days=args.min_days,
    )

    print(f"\n📊 Generated NegBin fit analysis for {args.count_type}")
