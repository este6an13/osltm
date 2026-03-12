from collections import defaultdict
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy.orm import Session

from src.constants.seed import SEED
from src.db.session_v2 import SessionLocal
from src.repo.v2.counts_15min.models import Counts15Min
from src.repo.v2.stations.repository import StationRepository

# ------------------------------------------------------------
# --- Distribution Simulations
# ------------------------------------------------------------


def simulate_poisson_series(lambdas: np.ndarray, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    return rng.poisson(lambdas)


def simulate_nb_15min_from_per_min(
    lambdas_per_min: np.ndarray,
    dispersion_per_min: np.ndarray,
    seed: Optional[int] = None,
):
    """
    Simulate Negative Binomial totals for each 15-min window by:
        - drawing 15 independent NB samples per window (one per-minute),
        - summing them to produce a 15-min total.

    lambdas_per_min: array-like shape (n_intervals,) -> mean per minute for each interval
    dispersion_per_min: array-like shape (n_intervals,) -> r (dispersion) per minute for each interval
    Returns: array shape (n_intervals,) of simulated 15-min totals
    """
    rng = np.random.default_rng(seed)
    lambdas_per_min = np.asarray(lambdas_per_min)
    dispersion_per_min = np.asarray(dispersion_per_min)

    n_intervals = lambdas_per_min.size
    # protect against tiny/zero dispersion
    dispersion_per_min = np.maximum(dispersion_per_min, 1e-6)

    # convert mean+dispersion -> NB param p: p = r / (r + lambda)
    p = dispersion_per_min / (dispersion_per_min + lambdas_per_min)
    # shape for sampling: (15 minutes, n_intervals)
    # numpy negative_binomial accepts (n, p) with n possibly float (works the same as earlier version)
    samples = rng.negative_binomial(
        dispersion_per_min.reshape(1, n_intervals),
        p.reshape(1, n_intervals),
        size=(15, n_intervals),
    )
    # sum 15 per-minute draws to make the 15-min aggregated count
    totals = samples.sum(axis=0)
    return totals


def simulate_nb_15min_from_per_min_vectorized(
    lambdas_per_min: np.ndarray,
    dispersion_per_min: np.ndarray,
    seed: Optional[int] = None,
):
    # Wrapper to keep naming consistent; same as above (kept for clarity)
    return simulate_nb_15min_from_per_min(lambdas_per_min, dispersion_per_min, seed)


# ------------------------------------------------------------
# --- Dispersion Estimation (per time window)
# ------------------------------------------------------------


def estimate_dispersion_from_mu_var(
    mu_by_day: list[np.ndarray], var_by_day: list[np.ndarray]
):
    """
    Estimate dispersion (r) per time interval using method-of-moments applied per-day:
        For each day and interval: r_day = μ_day^2 / (var_day - μ_day)  [only if var_day > μ_day]
    Then aggregate across days (median) to produce r per interval.

    Inputs:
        mu_by_day: list of 1-D arrays (per-interval μ_day = count_in/15) for each day
        var_by_day: list of 1-D arrays (per-interval var_day = variance_in_1min) for each day

    Returns:
        r_per_interval: 1-D array (n_intervals,) with a finite r for each interval (median across days).
    """
    if not mu_by_day:
        return np.array([])

    arr_mu = np.stack(mu_by_day)  # shape (n_days, n_intervals)
    arr_var = np.stack(var_by_day)

    # per-day, per-interval r_day (nan when not overdispersed)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_day = (arr_mu**2) / (arr_var - arr_mu)

    # treat values where var <= mu as NaN (Poisson-like)
    r_day = np.where(arr_var > arr_mu, r_day, np.nan)

    # compute median across days, ignoring NaNs
    r_median = np.nanmedian(r_day, axis=0)

    # If some intervals are all NaN (no day overdispersed), replace by global median of finite values
    finite_mask = np.isfinite(r_median)
    if not finite_mask.all():
        global_median = (
            np.nanmedian(r_median[finite_mask]) if finite_mask.any() else np.nan
        )
        # if still nan (no finite values anywhere), fallback to a large number approximating Poisson
        if np.isnan(global_median):
            global_median = 1e6
        r_median = np.where(np.isfinite(r_median), r_median, global_median)

    # final safety clamp
    r_median = np.maximum(r_median, 1e-6)
    return r_median


# ------------------------------------------------------------
# --- Performance / Metrics
# ------------------------------------------------------------


def evaluate_performance(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = ""
) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = (
        (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any()
        else np.nan
    )
    r2 = r2_score(y_true, y_pred)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}
    print(f"\n📈 Performance ({label}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    return metrics


# ------------------------------------------------------------
# --- Main: simulate_average_station_flows (v2 using counts_15min)
# ------------------------------------------------------------


def simulate_average_station_flows_v2(
    station_code: str,
    date_type: str,
    date_str: str,
    mode: str = "both",
    seed: Optional[int] = SEED,
    dist: str = "poisson",  # "poisson", "nb-global", or "nb-local"
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Simulate average flows per station using the v2 counts_15min table (count_in, variance_in_1min, count_out).
    - For IN: uses count_in and variance_in_1min (NB uses per-minute dispersion computed from these).
    - For OUT: defaults to Poisson (we have no per-minute variance for checkouts).
    """

    session: Session = SessionLocal()
    station_repo = StationRepository(session)

    # --- Validate station ---
    station = station_repo.get_station_by_code(station_code)
    if not station:
        print(f"⚠️ Station with code {station_code} not found.")
        session.close()
        return

    # parse comparison date
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year, month, day = date.year, date.month, date.day

    # Build base query
    q = session.query(Counts15Min).filter(
        Counts15Min.station_id == station.id,
        Counts15Min.date_type == date_type,
        Counts15Min.time >= 400,
        Counts15Min.time <= 2300,
    )

    # apply date range filters if provided
    if start_date:
        sd = datetime.strptime(start_date, "%Y-%m-%d")
        q = q.filter(
            (Counts15Min.year, Counts15Min.month, Counts15Min.day)
            >= (sd.year, sd.month, sd.day)
        )
    if end_date:
        ed = datetime.strptime(end_date, "%Y-%m-%d")
        q = q.filter(
            (Counts15Min.year, Counts15Min.month, Counts15Min.day)
            <= (ed.year, ed.month, ed.day)
        )
    if start_date or end_date:
        print(
            f"📅 Restricting to date range: {start_date or 'beginning'} → {end_date or 'end'}"
        )

    rows = q.order_by(
        Counts15Min.year, Counts15Min.month, Counts15Min.day, Counts15Min.time
    ).all()
    if not rows:
        print(
            f"⚠️ No counts found for {station_code} ({station.name}) and date_type '{date_type}'."
        )
        session.close()
        return

    # --- Group per-day and per-time window, collecting count_in, var_in, count_out
    daily = defaultdict(
        lambda: defaultdict(lambda: {"count_in": 0, "var_in": 0.0, "count_out": 0})
    )
    for r in rows:
        key = (r.year, r.month, r.day)
        # count_in may be None if missing — coerce to 0
        daily[key][r.time]["count_in"] = int(r.count_in or 0)
        # variance could be None; coerce to 0.0 — but will result in NaN dispersions later
        daily[key][r.time]["var_in"] = float(r.variance_in_1min or 0.0)
        daily[key][r.time]["count_out"] = int(r.count_out or 0)

    times = sorted({t for d in daily.values() for t in d.keys()})
    # Build per-day arrays for IN: μ_day (per-minute) and var_day (per-minute variance)
    mu_by_day = []
    var_by_day = []
    out_by_day = []  # per-day arrays of 15-min OUT counts (for Poisson avg/out)
    for day_vals in daily.values():
        mu_arr = np.array(
            [(day_vals[t]["count_in"] / 15.0) if t in day_vals else 0.0 for t in times]
        )
        var_arr = np.array(
            [day_vals[t]["var_in"] if t in day_vals else 0.0 for t in times]
        )
        out_arr = np.array(
            [day_vals[t]["count_out"] if t in day_vals else 0 for t in times]
        )
        mu_by_day.append(mu_arr)
        var_by_day.append(var_arr)
        out_by_day.append(out_arr)

    # Average curves (IN: average across days of per-minute means, then convert to 15-min mean; OUT: average 15-min totals)
    avg_mu_per_min = np.mean(np.stack(mu_by_day), axis=0)  # mean per-minute across days
    avg_count_in_per_15 = avg_mu_per_min * 15.0
    avg_count_out_per_15 = np.mean(np.stack(out_by_day), axis=0)

    # --- Estimate dispersion (r) per time for IN, if NB requested ---
    r_in = None
    if dist.startswith("nb"):
        r_in = estimate_dispersion_from_mu_var(
            mu_by_day, var_by_day
        )  # r measured per-minute
        if dist == "nb-global":
            # collapse to scalar
            r_in = np.nanmedian(r_in)
            # ensure we return array same shape when used later
            r_in = np.full_like(avg_mu_per_min, r_in, dtype=float)

    # --- Fetch actual counts for the comparison date ---
    actual_rows = (
        session.query(Counts15Min)
        .filter(
            Counts15Min.station_id == station.id,
            Counts15Min.year == year,
            Counts15Min.month == month,
            Counts15Min.day == day,
            Counts15Min.time >= 400,
            Counts15Min.time <= 2300,
        )
        .order_by(Counts15Min.time.asc())
        .all()
    )

    actual_times = []
    actual_count_in = np.array([])
    actual_count_out = np.array([])
    if actual_rows:
        actual_times = [r.time for r in actual_rows]
        actual_count_in = np.array([int(r.count_in or 0) for r in actual_rows])
        actual_count_out = np.array([int(r.count_out or 0) for r in actual_rows])

    # --- Simulation ---
    rng_seed = seed

    # IN simulation:
    sim_in = None
    if mode in ("in", "both"):
        if dist == "poisson":
            # Poisson at 15-min level (fast)
            sim_in = simulate_poisson_series(avg_count_in_per_15, rng_seed)
            dist_label_in = "Poisson"
        elif dist in ("nb-global", "nb-local"):
            # simulate per-minute NB and sum 15 samples per interval
            sim_in = simulate_nb_15min_from_per_min_vectorized(
                avg_mu_per_min, r_in, rng_seed
            )
            dist_label_in = (
                "NegBin (global)" if dist == "nb-global" else "NegBin (time-varying)"
            )
        else:
            raise ValueError(f"Unknown dist={dist}")

    # OUT simulation: default to Poisson (no per-minute variance available)
    sim_out = None
    if mode in ("out", "both"):
        sim_out = simulate_poisson_series(avg_count_out_per_15, rng_seed)
        dist_label_out = "Poisson (OUT)"

    # --- Estimate dispersion summary (print stats) ---
    # For IN: count overdispersed intervals based on per-day var > mu
    arr_mu = np.stack(mu_by_day)
    arr_var = np.stack(var_by_day)
    over_mask = (arr_var > arr_mu).any(
        axis=0
    )  # if any day shows var>mu we mark window as overdispersed
    num_over = int(over_mask.sum())
    total_w = int(over_mask.size)
    print(
        f"🧮 IN-dispersion summary: {num_over} / {total_w} windows overdispersed "
        f"({num_over / total_w:.1%}), {total_w - num_over} Poisson-like"
    )

    # --- Evaluate performance against actuals (if provided) ---
    if actual_rows:
        if mode in ("in", "both") and sim_in is not None:
            _ = evaluate_performance(actual_count_in, sim_in, label="IN")
        if mode in ("out", "both") and sim_out is not None:
            _ = evaluate_performance(actual_count_out, sim_out, label="OUT")

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    if mode in ("in", "both") and sim_in is not None:
        plt.plot(times, sim_in, "b-", label=f"Simulated IN ({dist_label_in})")
        if actual_times:
            plt.plot(
                actual_times, actual_count_in, "c--", label=f"Actual IN ({date_str})"
            )
    if mode in ("out", "both") and sim_out is not None:
        plt.plot(times, sim_out, "r-", label=f"Simulated OUT ({dist_label_out})")
        if actual_times:
            plt.plot(
                actual_times, actual_count_out, "m--", label=f"Actual OUT ({date_str})"
            )

    plt.title(
        f"Simulation vs Actual for {station.name} ({station.code})\n"
        f"Date type: {date_type} | Comparison date: {date_str}"
    )
    plt.xlabel("Time of day (HHMM, interval start)")
    plt.ylabel("Counts per 15-min interval")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    step = max(1, len(times) // 12)
    plt.xticks(
        ticks=times[::step],
        labels=[f"{t // 100:02d}:{t % 100:02d}" for t in times[::step]],
        rotation=45,
    )

    plt.show()
    session.close()


if __name__ == "__main__":
    # Example
    simulate_average_station_flows_v2(
        station_code="07107",
        date_type="WD",
        date_str="2025-10-16",
        mode="both",
        dist="nb-local",
        seed=SEED,
        start_date="2024-09-01",
        end_date="2025-10-10",
    )
