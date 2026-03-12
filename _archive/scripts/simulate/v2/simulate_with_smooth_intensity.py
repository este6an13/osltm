from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy.orm import Session

from src.db.session_v2 import SessionLocal
from src.repo.v2.counts_15min.models import Counts15Min
from src.repo.v2.stations.repository import StationRepository
from src.scripts.simulate.v2.compute_intensity_from_data import (
    estimate_intensity_then_mvf,
)

# ------------------------------------------------------------
# Utility conversions
# ------------------------------------------------------------


def hhmm_to_minutes(t):
    """Convert HHMM (e.g. 1630) to minutes since midnight."""
    return (t // 100) * 60 + (t % 100)


def minutes_to_hhmm(m):
    """Convert minutes since midnight to HHMM format."""
    h = m // 60
    mm = m % 60
    return h * 100 + mm


# ------------------------------------------------------------
# Performance metrics
# ------------------------------------------------------------


def evaluate_performance(y_true, y_pred, label=""):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true > 0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.any()
        else np.nan
    )
    r2 = r2_score(y_true, y_pred)

    print(f"\n📈 Performance ({label}):")
    print(f"  MAE : {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²  : {r2:.3f}")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}


# ------------------------------------------------------------
# Main model evaluation function
# ------------------------------------------------------------


def evaluate_intensity_model(
    station_code: str,
    date_type: str,
    comparison_date: str,
    smoothing_factor: float = 10.0,
    end_date: str = "2025-09-30",
):
    """
    - Calls estimate_intensity_then_mvf(), obtaining λ(t) at 5-min resolution.
    - Converts λ(t) to predicted 15-min counts via integration.
    - Compares them to actual DB 15-min counts.
    - Plots everything aligned.
    """

    session: Session = SessionLocal()
    station_repo = StationRepository(session)

    # --- Validate station ---
    station = station_repo.get_station_by_code(station_code)
    if not station:
        print(f"⚠️ Station {station_code} not found")
        return

    # --- Call your intensity estimation ---
    result = estimate_intensity_then_mvf(
        station_code=station_code,
        date_type=date_type,
        end_date=end_date,
        smoothing_factor=smoothing_factor,
    )

    # λ(t) is at 5-minute resolution
    t_smooth = np.array(result["time_smooth"])  # minutes since midnight (240–1380)
    lambda_in = np.array(result["lambda_in"])
    lambda_out = np.array(result["lambda_out"])

    # --- Fetch 15-min actual DB rows ---
    d = datetime.strptime(comparison_date, "%Y-%m-%d")
    rows = (
        session.query(Counts15Min)
        .filter(
            Counts15Min.station_id == station.id,
            Counts15Min.year == d.year,
            Counts15Min.month == d.month,
            Counts15Min.day == d.day,
            Counts15Min.time >= 400,
            Counts15Min.time <= 2300,
        )
        .order_by(Counts15Min.time)
        .all()
    )

    if not rows:
        print(f"⚠️ No actual rows found for date {comparison_date}")
        return

    actual_times_hhmm = np.array([r.time for r in rows])
    actual_in = np.array([int(r.count_in or 0) for r in rows])
    actual_out = np.array([int(r.count_out or 0) for r in rows])

    # Convert actual times to minutes since midnight
    actual_times_min = np.array([hhmm_to_minutes(t) for t in actual_times_hhmm])

    # ------------------------------------------------------------
    # Convert λ(t) to predicted 15-min totals (integration)
    # ------------------------------------------------------------

    # Δt = 5 minutes (your intensity sampling)
    dt = np.mean(np.diff(t_smooth))

    pred_in_15 = []
    pred_out_15 = []

    for start_min in actual_times_min:
        end_min = start_min + 15  # 15-min window

        mask = (t_smooth >= start_min) & (t_smooth < end_min)

        pred_in_15.append(np.sum(lambda_in[mask] * dt))
        pred_out_15.append(np.sum(lambda_out[mask] * dt))

    pred_in_15 = np.array(pred_in_15)
    pred_out_15 = np.array(pred_out_15)

    # ------------------------------------------------------------
    # PERFORMANCE
    # ------------------------------------------------------------
    evaluate_performance(actual_in, pred_in_15, label="IN")
    evaluate_performance(actual_out, pred_out_15, label="OUT")

    # ------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------

    # Convert smooth time to HHMM for plotting only
    t_smooth_hhmm = np.array([minutes_to_hhmm(m) for m in t_smooth])

    plt.figure(figsize=(13, 7))

    # Actual 15-min data
    plt.plot(actual_times_hhmm, actual_in, "c--", label="Actual IN (15m)")
    plt.plot(actual_times_hhmm, actual_out, "m--", label="Actual OUT (15m)")

    # Predicted 15-min data
    plt.plot(actual_times_hhmm, pred_in_15, "bo-", label="Predicted IN (15m)")
    plt.plot(actual_times_hhmm, pred_out_15, "ro-", label="Predicted OUT (15m)")

    plt.title(
        f"Estimated Intensity vs Actual 15-min Counts\n"
        f"{station.name} ({station.code}) — {date_type}, Comparison = {comparison_date}"
    )
    plt.xlabel("Time (HHMM)")
    plt.ylabel("Counts per 15 min")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Clean x-axis labels
    step = max(1, len(t_smooth_hhmm) // 12)
    plt.xticks(
        ticks=t_smooth_hhmm[::step],
        rotation=45,
    )

    plt.tight_layout()
    plt.show()

    session.close()


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    evaluate_intensity_model(
        station_code="07107",
        date_type="WD",
        comparison_date="2025-10-16",
        smoothing_factor=10.0,
        end_date="2025-09-30",
    )
