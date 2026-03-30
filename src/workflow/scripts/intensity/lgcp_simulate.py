"""
LGCP Simulation — Sample synthetic arrivals from the fitted Cox process

For each (station, day_type), draws from:
    z ~ N(μ, C_smooth)          GP prior (day-level intensity draw)
    N_k ~ Poisson(exp(z_k))     bin counts
    τ_{k,i} ~ Uniform(t_k, t_{k+1})  individual timestamps within bin

Outputs:
  1) Raw event CSV (one row per arrival):
       Fecha_Transaccion, Estacion_Parada
     Same format as data/check_ins/daily/*.csv — plugs directly into the pipeline.

  2) Binned count CSV (one row per simulated day):
       year, month, day, date_type, station_code, station_name, t_400, ...
     Same format as load_data() returns — for downstream analysis.

Uses Phase 2 kernel params (σ², ℓ) and Phase 3 posterior μ(t).
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.workflow.data_loader import load_data, load_persisted_data

DATE_TYPE_LABELS = {"WD": "Weekday", "SA": "Saturday", "SU": "Sunday", "HO": "Holiday"}


# ── Kernels (same as lgcp_bayesian.py) ────────────────────────────────────────


def kernel_se(t: np.ndarray, sigma2: float, ell: float, eta2: float) -> np.ndarray:
    diffs = t[:, None] - t[None, :]
    return sigma2 * np.exp(-0.5 * (diffs / ell) ** 2) + eta2 * np.eye(len(t))


def kernel_matern32(
    t: np.ndarray, sigma2: float, ell: float, eta2: float
) -> np.ndarray:
    diffs = np.abs(t[:, None] - t[None, :])
    r = np.sqrt(3) * diffs / ell
    return sigma2 * (1 + r) * np.exp(-r) + eta2 * np.eye(len(t))


# ── Time helpers ──────────────────────────────────────────────────────────────


def time_int_to_hours(t_int: int) -> float:
    return t_int // 100 + (t_int % 100) / 60.0


def hours_to_datetime(base_date: datetime, hours: float) -> datetime:
    """Convert fractional hours to a datetime on base_date."""
    h = int(hours)
    remainder = (hours - h) * 60
    m = int(remainder)
    s = int((remainder - m) * 60)
    return base_date.replace(hour=h, minute=m, second=s, microsecond=0)


def sort_time_columns(cols: list[str]) -> list[str]:
    return sorted(cols, key=lambda c: int(c.replace("t_", "")))


# ── Core simulation ──────────────────────────────────────────────────────────


def simulate_one_day(
    mu: np.ndarray,
    L: np.ndarray,
    time_hours: np.ndarray,
    bin_width_hours: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one day of arrivals from the LGCP.

    Returns:
        counts: (K,) array of Poisson counts per bin
        arrival_hours: 1-D array of arrival times (fractional hours)
    """
    K = len(mu)

    # 1) Draw log-intensity from GP prior
    eps = rng.standard_normal(K)
    z = mu + L @ eps  # z ~ N(μ, C_smooth)

    # 2) Poisson counts per bin
    lam = np.exp(z)
    counts = rng.poisson(lam).astype(int)

    # 3) Uniform timestamps within each bin
    arrivals = []
    for k in range(K):
        if counts[k] > 0:
            t_start = time_hours[k]
            t_end = t_start + bin_width_hours
            times = rng.uniform(t_start, t_end, size=counts[k])
            arrivals.append(times)

    arrival_hours = np.concatenate(arrivals) if arrivals else np.array([])
    arrival_hours.sort()

    return counts, arrival_hours


# ── Main ──────────────────────────────────────────────────────────────────────


def run_lgcp_simulate(
    n_days: int = 10,
    count_type: str = "checkins",
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes: Optional[list[str]] = None,
    phase2_dir: Optional[Path] = None,
    base_date: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """
    Simulate synthetic arrivals from the fitted LGCP.

    Args:
        n_days: Number of synthetic days to generate per (station, day_type).
        base_date: Starting date for synthetic timestamps (YYYYMMDD).
                   Defaults to '20260101'.
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
    bin_width_hours = time_step / 60.0

    persistence_dir = Path(step4.get("persistence_dir", "src/workflow/data"))
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)
    if sampled_stations:
        available = [s["code"] for s in sampled_stations]
        station_codes = (
            [sc for sc in station_codes if sc in available]
            if station_codes
            else available
        )
    elif station_codes is None:
        raise ValueError("No station codes available.")

    # ── Load Phase 2 kernel params ────────────────────────────────────────
    if phase2_dir is None:
        phase2_dir = Path("src/workflow/results/lgcp_twostage")
    phase2_dir = Path(phase2_dir)
    kernel_df = pd.read_csv(phase2_dir / f"lgcp_kernel_params_{count_type}.csv")
    kernel_df["station_code"] = kernel_df["station_code"].astype(str).str.zfill(5)
    kernel_sel = kernel_df[kernel_df["is_selected"] == True].copy()

    # ── Load count data (for μ = mean log-intensity) ──────────────────────
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

    # Time grid in hours
    time_hours = np.array([time_int_to_hours(int(c.replace("t_", ""))) for c in time_cols])

    if output_dir is None:
        output_dir = Path("src/workflow/results/lgcp_simulate")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_date is None:
        base_date = "20260101"
    base_dt = datetime.strptime(base_date, "%Y%m%d")

    rng = np.random.default_rng(seed)

    # ── Simulate per (station, day_type) ──────────────────────────────────
    all_events: list[dict] = []
    all_binned: list[dict] = []

    for _, krow in kernel_sel.iterrows():
        station_code = krow["station_code"]
        day_type = krow["day_type"]
        kernel_name = krow["kernel"]
        sigma2 = krow["sigma2"]
        ell = krow["ell_hours"]

        if station_code not in station_codes:
            continue

        dt_label = DATE_TYPE_LABELS.get(day_type, day_type)

        # Get observed data for this (s, d)
        sdf = df[
            (df["station_code"] == station_code) & (df["date_type"] == day_type)
        ]
        if len(sdf) < 5:
            continue

        counts_all = sdf[time_cols].fillna(0).values.astype(float)
        mean_counts = counts_all.mean(axis=0)
        mu = np.log(mean_counts + 0.5)

        # Get station name
        station_name = (
            sdf["station_name"].iloc[0] if "station_name" in sdf.columns
            else station_code
        )

        # Build smooth GP kernel (without nugget) and Cholesky factor
        kernel_fn = kernel_se if kernel_name == "SE" else kernel_matern32
        C_smooth = kernel_fn(time_hours, sigma2, ell, 0.0)
        C_smooth += 1e-6 * np.eye(K)

        try:
            L = np.linalg.cholesky(C_smooth)
        except np.linalg.LinAlgError:
            # Fallback: add more regularisation
            C_smooth += 1e-4 * np.eye(K)
            L = np.linalg.cholesky(C_smooth)

        print(f"\n{station_code}/{dt_label}: simulating {n_days} days "
              f"(kernel={kernel_name}, σ²={sigma2:.4f}, ℓ={ell:.2f}h)")

        # Station string for raw CSV (matching existing format)
        estacion_str = f"({station_code}) {station_name}"

        for j in range(n_days):
            sim_date = base_dt + timedelta(days=j)

            counts, arrival_hours = simulate_one_day(
                mu, L, time_hours, bin_width_hours, rng
            )

            total = int(counts.sum())
            print(f"  Day {j+1}: {total} arrivals")

            # ── Raw events ────────────────────────────────────────────────
            for t_h in arrival_hours:
                dt = hours_to_datetime(sim_date, float(t_h))
                all_events.append({
                    "Fecha_Transaccion": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "Estacion_Parada": estacion_str,
                })

            # ── Binned row ────────────────────────────────────────────────
            row = {
                "year": sim_date.year,
                "month": sim_date.month,
                "day": sim_date.day,
                "date_type": day_type,
                "station_code": station_code,
                "station_name": station_name,
            }
            for k, col in enumerate(time_cols):
                row[col] = int(counts[k])
            all_binned.append(row)

    # ── Save raw events CSV ───────────────────────────────────────────────
    events_df = pd.DataFrame(all_events)
    events_csv = output_dir / f"lgcp_simulated_events_{count_type}.csv"
    events_df.to_csv(events_csv, index=False)
    print(f"\nSaved {len(events_df)} simulated events: {events_csv}")

    # ── Save binned counts CSV ────────────────────────────────────────────
    binned_df = pd.DataFrame(all_binned)
    binned_csv = output_dir / f"lgcp_simulated_binned_{count_type}.csv"
    binned_df.to_csv(binned_csv, index=False)
    print(f"Saved {len(binned_df)} simulated day-profiles: {binned_csv}")

    # ── Quick validation plot ─────────────────────────────────────────────
    _plot_simulation_comparison(
        df, binned_df, time_cols, time_hours, output_dir, count_type
    )

    return {"events_df": events_df, "binned_df": binned_df}


def _plot_simulation_comparison(
    observed_df, simulated_df, time_cols, time_hours, output_dir, count_type
):
    """Plot observed vs simulated mean profiles per (station, day_type)."""
    import matplotlib.pyplot as plt

    pairs = simulated_df.groupby(["station_code", "date_type"]).size().index.tolist()
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    cols = min(4, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (sc, dt) in enumerate(pairs):
        ax = axes[idx // cols][idx % cols]

        obs = observed_df[
            (observed_df["station_code"] == sc) & (observed_df["date_type"] == dt)
        ][time_cols].fillna(0)
        sim = simulated_df[
            (simulated_df["station_code"] == sc) & (simulated_df["date_type"] == dt)
        ][time_cols].fillna(0)

        obs_mean = obs.mean().values
        sim_mean = sim.mean().values
        sim_std = sim.std().values

        ax.plot(time_hours, obs_mean, "k-", lw=2, label="Observed mean")
        ax.plot(time_hours, sim_mean, "b-", lw=1.5, alpha=0.8, label="Simulated mean")
        ax.fill_between(
            time_hours,
            sim_mean - 2 * sim_std, sim_mean + 2 * sim_std,
            alpha=0.15, color="blue", label="Sim ±2σ",
        )

        dt_label = DATE_TYPE_LABELS.get(dt, dt)
        ax.set_title(f"{sc} {dt_label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Counts / bin")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_pairs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"LGCP Simulation vs Observed ({count_type.capitalize()})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fname = output_dir / f"lgcp_simulation_comparison_{count_type}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Saved comparison plot: {fname}")
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate arrivals from the fitted LGCP model"
    )
    parser.add_argument("--n_days", type=int, default=10,
                        help="Days to simulate per (station, day_type)")
    parser.add_argument("--count_type", default="checkins", choices=["checkins", "checkouts"])
    parser.add_argument("--output_dir", default="src/workflow/results/lgcp_simulate")
    parser.add_argument("--params", default="src/workflow/params.json")
    parser.add_argument("--stations", nargs="+", default=None)
    parser.add_argument("--phase2_dir", default="src/workflow/results/lgcp_twostage")
    parser.add_argument("--base_date", default="20260101",
                        help="Start date for synthetic timestamps (YYYYMMDD)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_lgcp_simulate(
        n_days=args.n_days,
        count_type=args.count_type,
        output_dir=Path(args.output_dir),
        params_path=Path(args.params),
        station_codes=args.stations,
        phase2_dir=Path(args.phase2_dir),
        base_date=args.base_date,
        seed=args.seed,
    )
