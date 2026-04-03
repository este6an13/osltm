"""
Simulation Performance Comparison — LGCP vs Hawkes × {checkins, checkouts}

Loads the observed 15-min binned data and each model's simulated binned
data, then computes per-(station, day_type) metrics:

  1.  MAE   — mean absolute error of profile means
  2.  RMSE  — root-mean-squared error of profile means
  3.  MAPE  — mean absolute percentage error (bins with obs > 0)
  4.  Pearson r  — correlation between observed & simulated mean profiles
  5.  Total-count ratio  — sim_total / obs_total  (calibration)
  6.  Wasserstein dist  — Earth Mover's Distance on normalised profiles
  7.  Coverage  — fraction of observed mean bins inside sim ±2σ band

Outputs:
  - simulation_comparison_metrics.csv  — one row per (model, ct, station, dt)
  - simulation_comparison_summary.csv  — one row per (model, ct)
  - simulation_comparison_heatmap.png  — faceted heatmap of key metrics
  - simulation_comparison_bars.png     — bar chart model × count_type
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.workflow.data_loader import load_data, load_persisted_data

DATE_TYPE_LABELS = {"WD": "Weekday", "SA": "Saturday", "SU": "Sunday", "HO": "Holiday"}
DATE_TYPE_ORDER = ["WD", "SA", "SU", "HO"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def sort_time_columns(cols: list[str]) -> list[str]:
    return sorted(cols, key=lambda c: int(c.replace("t_", "")))


def wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein-1 distance between two non-negative vectors (treated as unnormalised densities)."""
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return np.nan
    p_norm = p / p_sum
    q_norm = q / q_sum
    cdf_p = np.cumsum(p_norm)
    cdf_q = np.cumsum(q_norm)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


# ── Per-pair metrics ─────────────────────────────────────────────────────────


def compute_pair_metrics(
    obs_counts: np.ndarray,
    sim_counts: np.ndarray,
) -> dict:
    """
    Compute comparison metrics for one (station, day_type) pair.

    Args:
        obs_counts: (J_obs, K) array of observed day profiles
        sim_counts: (J_sim, K) array of simulated day profiles

    Returns:
        dict of metric name → value
    """
    obs_mean = obs_counts.mean(axis=0)  # (K,)
    sim_mean = sim_counts.mean(axis=0)
    sim_std = sim_counts.std(axis=0, ddof=1) if len(sim_counts) > 1 else np.zeros_like(sim_mean)

    K = len(obs_mean)

    # MAE
    mae = float(np.mean(np.abs(obs_mean - sim_mean)))

    # RMSE
    rmse = float(np.sqrt(np.mean((obs_mean - sim_mean) ** 2)))

    # MAPE (only bins with obs > 0)
    nonzero = obs_mean > 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((obs_mean[nonzero] - sim_mean[nonzero]) / obs_mean[nonzero])) * 100)
    else:
        mape = np.nan

    # Pearson correlation
    if np.std(obs_mean) > 0 and np.std(sim_mean) > 0:
        pearson_r, _ = sp_stats.pearsonr(obs_mean, sim_mean)
        pearson_r = float(pearson_r)
    else:
        pearson_r = np.nan

    # Total count ratio
    obs_total = obs_mean.sum()
    sim_total = sim_mean.sum()
    total_ratio = float(sim_total / obs_total) if obs_total > 0 else np.nan

    # Wasserstein distance on normalised profiles
    wd = wasserstein_1d(obs_mean, sim_mean)

    # Coverage: fraction of bins where obs_mean falls within sim ±2σ
    lo = sim_mean - 2 * sim_std
    hi = sim_mean + 2 * sim_std
    coverage = float(np.mean((obs_mean >= lo) & (obs_mean <= hi)))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "pearson_r": pearson_r,
        "total_count_ratio": total_ratio,
        "wasserstein": wd,
        "coverage_2sigma": coverage,
        "obs_days": len(obs_counts),
        "sim_days": len(sim_counts),
        "n_bins": K,
    }


# ── Load simulated binned data ──────────────────────────────────────────────


def load_sim_binned(model: str, count_type: str, results_root: Path) -> Optional[pd.DataFrame]:
    """Load a simulated binned CSV if it exists."""
    if model == "lgcp":
        path = results_root / "lgcp_simulate" / f"lgcp_simulated_binned_{count_type}.csv"
    elif model == "hawkes":
        path = results_root / "hawkes_simulate" / f"hawkes_simulated_binned_{count_type}.csv"
    else:
        return None

    if not path.exists():
        print(f"  ⚠️  Not found: {path}")
        return None

    df = pd.read_csv(path)
    df["station_code"] = df["station_code"].astype(str).str.zfill(5)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────


def run_simulation_comparison(
    params_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    results_root: Optional[Path] = None,
):
    # ── Load params & observed data ───────────────────────────────────────
    if params_path is None:
        params_path = Path("src/workflow/params.json")
    with open(params_path) as f:
        params = json.load(f)

    step4 = params.get("step4", {})
    time_min = step4.get("time_min", 400)
    time_max = step4.get("time_max", 2300)
    time_step = step4.get("time_step", 15)

    if results_root is None:
        results_root = Path("src/workflow/results")
    if output_dir is None:
        output_dir = results_root / "simulation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load observed data for both count types
    observed = {}
    for ct in ["checkins", "checkouts"]:
        print(f"📊 Loading observed {ct} data...")
        try:
            data = load_data(
                include_checkins=(ct == "checkins"),
                include_checkouts=(ct == "checkouts"),
                time_min=time_min, time_max=time_max, time_step=time_step,
            )
            observed[ct] = data[ct]
        except Exception as e:
            print(f"  ⚠️  Could not load {ct}: {e}")

    # ── Evaluate each (model, count_type) combination ─────────────────────
    models = ["lgcp", "hawkes"]
    count_types = ["checkins", "checkouts"]

    all_rows: list[dict] = []

    for model in models:
        for ct in count_types:
            if ct not in observed:
                continue

            sim_df = load_sim_binned(model, ct, results_root)
            if sim_df is None:
                continue

            obs_df = observed[ct]
            time_cols = sort_time_columns([c for c in obs_df.columns if c.startswith("t_")])

            # Iterate over (station, day_type) pairs present in simulation
            pairs = sim_df.groupby(["station_code", "date_type"]).size().index.tolist()

            for station_code, day_type in pairs:
                obs_slice = obs_df[
                    (obs_df["station_code"] == station_code) & (obs_df["date_type"] == day_type)
                ]
                sim_slice = sim_df[
                    (sim_df["station_code"] == station_code) & (sim_df["date_type"] == day_type)
                ]

                if len(obs_slice) < 3 or len(sim_slice) < 2:
                    continue

                obs_counts = obs_slice[time_cols].fillna(0).values.astype(float)
                sim_counts = sim_slice[time_cols].fillna(0).values.astype(float)

                metrics = compute_pair_metrics(obs_counts, sim_counts)

                station_name = (
                    obs_slice["station_name"].iloc[0]
                    if "station_name" in obs_slice.columns
                    else station_code
                )

                all_rows.append({
                    "model": model.upper(),
                    "count_type": ct,
                    "station_code": station_code,
                    "station_name": station_name,
                    "day_type": day_type,
                    **metrics,
                })

    # ── Build results DataFrame ───────────────────────────────────────────
    if not all_rows:
        print("❌ No model × count_type combinations had matched data.")
        return

    metrics_df = pd.DataFrame(all_rows)
    metrics_csv = output_dir / "simulation_comparison_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n💾 Saved per-pair metrics ({len(metrics_df)} rows): {metrics_csv}")

    # ── Summary table: aggregate per (model, count_type) ──────────────────
    agg_cols = ["mae", "rmse", "mape_pct", "pearson_r", "total_count_ratio",
                "wasserstein", "coverage_2sigma"]
    summary = (
        metrics_df.groupby(["model", "count_type"])[agg_cols]
        .agg(["mean", "median"])
    )
    # Flatten multi-level columns
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()

    summary_csv = output_dir / "simulation_comparison_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"💾 Saved summary table: {summary_csv}")

    # Print to console
    print("\n" + "=" * 80)
    print("SIMULATION PERFORMANCE COMPARISON")
    print("=" * 80)
    for _, row in summary.iterrows():
        print(f"\n  {row['model']} × {row['count_type']}")
        print(f"    MAE          mean={row['mae_mean']:.2f}   median={row['mae_median']:.2f}")
        print(f"    RMSE         mean={row['rmse_mean']:.2f}   median={row['rmse_median']:.2f}")
        print(f"    MAPE (%)     mean={row['mape_pct_mean']:.1f}   median={row['mape_pct_median']:.1f}")
        print(f"    Pearson r    mean={row['pearson_r_mean']:.4f}   median={row['pearson_r_median']:.4f}")
        print(f"    Count ratio  mean={row['total_count_ratio_mean']:.3f}   median={row['total_count_ratio_median']:.3f}")
        print(f"    Wasserstein  mean={row['wasserstein_mean']:.4f}   median={row['wasserstein_median']:.4f}")
        print(f"    Coverage 2σ  mean={row['coverage_2sigma_mean']:.1%}   median={row['coverage_2sigma_median']:.1%}")

    # ── Plotting ──────────────────────────────────────────────────────────
    _plot_comparison_bars(metrics_df, output_dir)
    _plot_comparison_heatmap(metrics_df, output_dir)

    print(f"\n✅ Simulation comparison complete.  Results in {output_dir}")
    return {"metrics_df": metrics_df, "summary": summary}


# ── Visualisations ───────────────────────────────────────────────────────────


def _plot_comparison_bars(metrics_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart: key metrics per (model × count_type)."""

    key_metrics = [
        ("mae", "MAE (counts/bin)", False),
        ("rmse", "RMSE (counts/bin)", False),
        ("pearson_r", "Pearson r", True),
        ("coverage_2sigma", "Coverage (2σ band)", True),
    ]

    fig, axes = plt.subplots(1, len(key_metrics), figsize=(5 * len(key_metrics), 5))

    groups = metrics_df.groupby(["model", "count_type"])
    group_labels = sorted(groups.groups.keys())
    x = np.arange(len(group_labels))
    colors = {"LGCP": "#2980b9", "HAWKES": "#e74c3c"}

    for ax, (col, ylabel, higher_better) in zip(axes, key_metrics):
        medians = [groups.get_group(g)[col].median() for g in group_labels]
        q25 = [groups.get_group(g)[col].quantile(0.25) for g in group_labels]
        q75 = [groups.get_group(g)[col].quantile(0.75) for g in group_labels]
        bar_colors = [colors.get(g[0], "#999") for g in group_labels]

        yerr_lo = [m - q for m, q in zip(medians, q25)]
        yerr_hi = [q - m for m, q in zip(medians, q75)]

        ax.bar(x, medians, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, medians, yerr=[yerr_lo, yerr_hi],
                     fmt="none", ecolor="black", capsize=4, linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{g[0]}\n{g[1]}" for g in group_labels],
            fontsize=9,
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        arrow = "↑" if higher_better else "↓"
        ax.set_title(f"{ylabel}\n(better {arrow})", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Simulation Quality: LGCP vs Hawkes",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fname = output_dir / "simulation_comparison_bars.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"📊 Saved bar chart: {fname}")
    plt.close()


def _plot_comparison_heatmap(metrics_df: pd.DataFrame, output_dir: Path):
    """Heatmap: one row per (station, day_type), columns = models, cell = Pearson r."""

    count_types = [ct for ct in ["checkins", "checkouts"] if ct in metrics_df["count_type"].values]
    n_ct = len(count_types)
    if n_ct == 0:
        return

    fig, axes = plt.subplots(1, n_ct, figsize=(8 * n_ct, 10), squeeze=False)

    for col_idx, ct in enumerate(count_types):
        ax = axes[0][col_idx]
        ct_df = metrics_df[metrics_df["count_type"] == ct].copy()

        if ct_df.empty:
            ax.set_title(f"{ct.capitalize()} — no data")
            continue

        # Sort day_types, then stations
        ct_df["dt_order"] = ct_df["day_type"].map({d: i for i, d in enumerate(DATE_TYPE_ORDER)})
        ct_df = ct_df.sort_values(["station_code", "dt_order"])
        ct_df["label"] = ct_df["station_code"] + " " + ct_df["day_type"].map(DATE_TYPE_LABELS)

        # Pivot: rows = label, columns = model, values = pearson_r
        pivot = ct_df.pivot_table(index="label", columns="model", values="pearson_r")
        # Sort by station order
        label_order = ct_df.drop_duplicates("label")["label"].tolist()
        pivot = pivot.reindex([l for l in label_order if l in pivot.index])

        if pivot.empty:
            continue

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
        )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.75 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

        ax.set_title(f"{ct.capitalize()}", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")

    fig.suptitle(
        "Profile Correlation (Pearson r) per Station × Day-Type",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    fname = output_dir / "simulation_comparison_heatmap.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"📊 Saved heatmap: {fname}")
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare simulation quality: LGCP vs Hawkes × {checkins, checkouts}"
    )
    parser.add_argument("--params", default="src/workflow/params.json")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--results_root", default="src/workflow/results")

    args = parser.parse_args()
    run_simulation_comparison(
        params_path=Path(args.params),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        results_root=Path(args.results_root),
    )
