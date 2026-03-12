from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sqlalchemy.orm import Session

from src.constants.seed import SEED
from src.db.session_v1 import SessionLocal
from src.repo.v1.estimates.models import Estimate
from src.repo.v1.stations.repository import StationRepository

# ------------------------------------------------------------
# --- Distribution Simulations
# ------------------------------------------------------------


def simulate_poisson_series(lambdas: np.ndarray, seed: int | None = None):
    rng = np.random.default_rng(seed)
    return rng.poisson(lambdas)


def simulate_negative_binomial_series(
    lambdas: np.ndarray, dispersion: np.ndarray, seed: int | None = None
):
    """
    Simulate counts using a Negative Binomial model.
    'dispersion' (r) can be a scalar or array (same length as lambdas).
    """
    rng = np.random.default_rng(seed)
    dispersion = np.asarray(dispersion)
    dispersion = np.maximum(dispersion, 1e-6)  # avoid zero

    # Convert mean-dispersion parameterization to NB (n, p)
    p = dispersion / (dispersion + lambdas)
    return rng.negative_binomial(dispersion, p)


# ------------------------------------------------------------
# --- Dispersion Estimation
# ------------------------------------------------------------


def estimate_dispersion_per_time(lambdas_by_day: list[np.ndarray]):
    """
    Estimate dispersion (r) per time interval using method-of-moments:
    Var(Y) = μ + μ^2 / r  =>  r = μ^2 / (Var - μ)

    Prints number of overdispersed vs Poisson-like windows.
    """
    arr = np.stack(lambdas_by_day)  # shape: (num_days, num_intervals)
    means = arr.mean(axis=0)
    variances = arr.var(axis=0)

    # Identify overdispersion
    over_mask = variances > means
    num_over = np.sum(over_mask)
    num_poisson_like = np.sum(~over_mask)
    total = len(means)

    print(
        f"🧮 Dispersion summary: "
        f"{num_over} / {total} windows overdispersed "
        f"({num_over / total:.1%}), "
        f"{num_poisson_like} / {total} Poisson-like "
        f"({num_poisson_like / total:.1%})"
    )

    # Compute r where overdispersed, otherwise inf (≈ Poisson)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(over_mask, (means**2) / (variances - means), np.inf)

    # Replace inf/nan with large finite value (approx Poisson)
    r = np.where(np.isfinite(r), r, np.nan)
    median_r = np.nanmedian(r)
    r = np.where(np.isnan(r), median_r, r)

    return r


# ------------------------------------------------------------
# --- Main Simulation Function
# ------------------------------------------------------------


def evaluate_performance(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = ""
) -> dict:
    """
    Compute performance metrics for simulation vs actual.
    Returns dict with MAE, RMSE, MAPE, and R².
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.any()
        else np.nan
    )
    r2 = r2_score(y_true, y_pred)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}
    print(f"\n📈 Performance ({label}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    return metrics


def simulate_average_station_flows(
    station_code: str,
    date_type: str,
    date_str: str,
    mode: str = "both",
    seed: int | None = SEED,
    dist: str = "poisson",  # "poisson", "nb-global", or "nb-local"
    start_date: str | None = None,  # e.g. "2025-09-01"
    end_date: str | None = None,  # e.g. "2025-10-31"
):
    """
    Simulate average flows per station and date_type using optional date range filtering.
    """

    session: Session = SessionLocal()
    station_repo = StationRepository(session)

    # --- Validate station ---
    station = station_repo.get_station_by_code(station_code)
    if not station:
        print(f"⚠️ Station with code {station_code} not found.")
        session.close()
        return

    # --- Parse main comparison date ---
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year, month, day = date.year, date.month, date.day

    # --- Parse optional date range ---
    date_filters = []
    if start_date:
        sd = datetime.strptime(start_date, "%Y-%m-%d")
        date_filters.append(
            tuple([Estimate.year, Estimate.month, Estimate.day])
            >= (sd.year, sd.month, sd.day)
        )
    if end_date:
        ed = datetime.strptime(end_date, "%Y-%m-%d")
        date_filters.append(
            tuple([Estimate.year, Estimate.month, Estimate.day])
            <= (ed.year, ed.month, ed.day)
        )

    # --- Base query ---
    query = session.query(Estimate).filter(
        Estimate.station_id == station.id,
        Estimate.date_type == date_type,
        Estimate.time >= 400,  # Start at 04:00
        Estimate.time <= 2300,  # End at 23:00
    )

    # --- Apply date range if provided ---
    if start_date or end_date:
        for f in date_filters:
            query = query.filter(f)
        print(
            f"📅 Restricting to date range: {start_date or 'beginning'} → {end_date or 'end'}"
        )

    estimates = query.order_by(
        Estimate.year, Estimate.month, Estimate.day, Estimate.time
    ).all()

    if not estimates:
        print(
            f"⚠️ No estimates found for {station_code} ({station.name}) and date_type '{date_type}'."
        )
        session.close()
        return

    # --- Group λs by day and time ---
    daily_dict = defaultdict(lambda: defaultdict(list))
    for e in estimates:
        key = (e.year, e.month, e.day)
        daily_dict[key][e.time].append(
            (e.estimated_lambda_in or 0, e.estimated_lambda_out or 0)
        )

    times = sorted({t for d in daily_dict.values() for t in d.keys()})
    daily_lambda_in, daily_lambda_out = [], []
    for day_vals in daily_dict.values():
        in_series = np.array(
            [np.mean(day_vals[t][0][0]) if t in day_vals else 0 for t in times]
        )
        out_series = np.array(
            [np.mean(day_vals[t][0][1]) if t in day_vals else 0 for t in times]
        )
        daily_lambda_in.append(in_series)
        daily_lambda_out.append(out_series)

    # --- Average curves ---
    avg_lambda_in = np.mean(daily_lambda_in, axis=0)
    avg_lambda_out = np.mean(daily_lambda_out, axis=0)

    # --- Estimate dispersion per time (for NB-local) ---
    if dist.startswith("nb"):
        r_in = estimate_dispersion_per_time(daily_lambda_in)
        r_out = estimate_dispersion_per_time(daily_lambda_out)
        if dist == "nb-global":
            r_in = np.nanmedian(r_in)
            r_out = np.nanmedian(r_out)

    # --- Fetch actual λs for the specific comparison date ---
    actual_estimates = (
        session.query(Estimate)
        .filter(
            Estimate.station_id == station.id,
            Estimate.year == year,
            Estimate.month == month,
            Estimate.day == day,
            Estimate.time >= 400,
            Estimate.time <= 2300,
        )
        .order_by(Estimate.time.asc())
        .all()
    )

    actual_times, actual_lambda_in, actual_lambda_out = [], [], []
    if actual_estimates:
        actual_times = [e.time for e in actual_estimates]
        actual_lambda_in = np.array(
            [e.estimated_lambda_in or 0 for e in actual_estimates]
        )
        actual_lambda_out = np.array(
            [e.estimated_lambda_out or 0 for e in actual_estimates]
        )

    # --- Simulation ---
    if dist == "poisson":
        sim_in = (
            simulate_poisson_series(avg_lambda_in, seed)
            if mode in ("in", "both")
            else None
        )
        sim_out = (
            simulate_poisson_series(avg_lambda_out, seed)
            if mode in ("out", "both")
            else None
        )
        dist_label = "Poisson"
    elif dist in ("nb-global", "nb-local"):
        sim_in = (
            simulate_negative_binomial_series(avg_lambda_in, r_in, seed)
            if mode in ("in", "both")
            else None
        )
        sim_out = (
            simulate_negative_binomial_series(avg_lambda_out, r_out, seed)
            if mode in ("out", "both")
            else None
        )
        dist_label = (
            "NegBin (global)" if dist == "nb-global" else "NegBin (time-varying)"
        )
    else:
        raise ValueError(f"Unknown dist={dist}")

    # --- Evaluate performance ---
    if len(actual_times) > 0:
        if mode in ("in", "both"):
            _ = evaluate_performance(actual_lambda_in, sim_in, label="IN")
        if mode in ("out", "both"):
            _ = evaluate_performance(actual_lambda_out, sim_out, label="OUT")

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    if mode in ("in", "both"):
        plt.plot(times, sim_in, "b-", label=f"Simulated IN ({dist_label})")
        if actual_times:
            plt.plot(
                actual_times, actual_lambda_in, "c--", label=f"Actual IN ({date_str})"
            )
    if mode in ("out", "both"):
        plt.plot(times, sim_out, "r-", label=f"Simulated OUT ({dist_label})")
        if actual_times:
            plt.plot(
                actual_times, actual_lambda_out, "m--", label=f"Actual OUT ({date_str})"
            )

    plt.title(
        f"{dist_label} Simulation vs Actual for {station.name} ({station.code})\n"
        f"Date type: {date_type} | Comparison date: {date_str}"
    )
    plt.xlabel("Time of day (HHMM, interval start)")
    plt.ylabel("Counts per interval")
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
    # Example usage
    simulate_average_station_flows(
        station_code="07107",  # U Nacional
        date_type="WD",  # e.g. "WD", "SAT", "HOL"
        date_str="2025-10-16",  # Comparison date
        mode="both",  # "in", "out", or "both"
        dist="nb-local",
        seed=SEED,
        start_date="2024-09-01",
        end_date="2025-10-10",
    )
