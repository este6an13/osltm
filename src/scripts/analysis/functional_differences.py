from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sqlalchemy.orm import Session

from src.db.session_v1 import SessionLocal
from src.repo.v1.estimates.models import Estimate

# ==========================================================
# 0. Data Loading
# ==========================================================


def _load_estimate_data(station_ids: list[int]) -> pd.DataFrame:
    """Load estimate records for selected stations from the database."""
    session: Session = SessionLocal()
    query = session.query(Estimate).filter(Estimate.station_id.in_(station_ids))
    df = pd.read_sql(query.statement, session.bind)
    session.close()
    return df


# ==========================================================
# 1. Helper: build daily functions
# ==========================================================


def _reshape_to_daily_functions(df: pd.DataFrame, lambda_col: str) -> pd.DataFrame:
    """
    Convert raw 15-min records into one vector per day (a discrete function).
    Each row = one day (station_id, day, date_type, month, day_of_week, λ_t1 ... λ_tN)
    """
    grouped = (
        df.groupby(["station_id", "day", "date_type", "month", "day_of_week"])[
            lambda_col
        ]
        .apply(list)
        .reset_index()
        .rename(columns={lambda_col: "lambda_curve"})
    )
    # Remove days with incomplete 96-point curves
    grouped["n_points"] = grouped["lambda_curve"].apply(len)
    grouped = grouped[grouped["n_points"] >= 80]  # tolerate some missing windows
    return grouped


def _align_and_interpolate_curves(
    curves: list[np.ndarray], target_len: int = 96
) -> np.ndarray:
    """
    Interpolate each curve to the same length (default: 96 time windows per day).
    Handles NaNs via linear interpolation; drops curves with too few valid points.
    """
    aligned = []
    for arr in curves:
        arr = np.array(arr, dtype=float)

        # Skip curves with almost no valid data
        if np.sum(~np.isnan(arr)) < 10:  # fewer than 10 valid points
            continue

        # Fill NaNs by interpolation (linear over valid indices)
        x = np.arange(len(arr))
        if np.any(np.isnan(arr)):
            valid = ~np.isnan(arr)
            arr = np.interp(x, x[valid], arr[valid])  # interpolate NaNs

        # Interpolate to common length
        n = len(arr)
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_len)
        arr_interp = np.interp(x_new, x_old, arr)
        aligned.append(arr_interp)

    if not aligned:
        return np.empty((0, target_len))

    return np.stack(aligned)


# ==========================================================
# 2. Functional correlation across groups
# ==========================================================


def functional_correlation_analysis(
    station_ids: List[int],
    lambda_type: Literal["in", "out"] = "in",
    group_by: Literal["date_type", "month", "day_of_week"] = "month",
    fixed_by: Literal["date_type", "month", "day_of_week"] | None = None,
    fixed_value: str | int | None = None,
    normalize: bool = True,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Compute average pairwise correlation between daily curves
    across groups (e.g., months or date types) for each station.

    If `fixed_by` and `fixed_value` are provided, data will first be filtered
    to include only rows where fixed_by == fixed_value. For example:
        fixed_by="date_type", fixed_value="HO" → only Holidays.
    """
    df = _load_estimate_data(station_ids)
    value_col = "estimated_lambda_in" if lambda_type == "in" else "estimated_lambda_out"
    daily = _reshape_to_daily_functions(df, value_col)

    results = []

    for station_id, g in daily.groupby("station_id"):
        if g.empty:
            continue

        # Optional filtering to fix one factor (e.g., only Holidays)
        if fixed_by and fixed_value is not None:
            g = g[g[fixed_by] == fixed_value]
            if g.empty:
                continue

        for group_name, sub in g.groupby(group_by):
            curves = []
            for row in sub.itertuples():
                arr = np.array(row.lambda_curve, dtype=float)
                if normalize:
                    total = np.nansum(arr)
                    if total > 0:
                        arr = arr / total
                curves.append(arr)

            if len(curves) < 2:
                continue

            # Align and clean curves
            curves = _align_and_interpolate_curves(curves)
            if len(curves) == 0:
                continue

            # Pairwise correlations
            n = len(curves)
            corr_vals = []
            for i in range(n):
                for j in range(i + 1, n):
                    if np.nanstd(curves[i]) == 0 or np.nanstd(curves[j]) == 0:
                        continue
                    c = np.corrcoef(curves[i], curves[j])[0, 1]
                    if not np.isnan(c):
                        corr_vals.append(c)

            if len(corr_vals) == 0:
                continue

            results.append(
                {
                    "station_id": station_id,
                    "group_by": group_by,
                    "fixed_by": fixed_by,
                    "fixed_value": fixed_value,
                    "group_label": group_name,
                    "n_days": n,
                    "mean_corr": np.nanmean(corr_vals),
                    "std_corr": np.nanstd(corr_vals),
                }
            )

            if plot:
                mean_curve = np.nanmean(curves, axis=0)
                plt.plot(mean_curve, label=f"{group_name} (n={n})")

        if plot:
            title_suffix = (
                f" (only {fixed_by}={fixed_value})" if fixed_by and fixed_value else ""
            )
            plt.title(
                f"Mean normalized λ_{lambda_type} curves by {group_by} — Station {station_id}{title_suffix}"
            )
            plt.xlabel("15-min interval")
            plt.ylabel("Normalized λ(t)")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)


# ==========================================================
# 3. Functional ANOVA
# ==========================================================


def functional_anova_analysis(
    station_ids: List[int],
    lambda_type: Literal["in", "out"] = "in",
    group_by: Literal["date_type", "month", "day_of_week"] = "month",
    fixed_by: Literal["date_type", "month", "day_of_week"] | None = None,
    fixed_value: str | int | None = None,
    normalize: bool = True,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Perform functional ANOVA comparing mean λ(t) curves across groups
    (e.g., months), while optionally holding another factor constant
    (e.g., only Holidays).

    If `fixed_by` and `fixed_value` are set, filters data to only that subset.
    """
    df = _load_estimate_data(station_ids)
    value_col = "estimated_lambda_in" if lambda_type == "in" else "estimated_lambda_out"
    daily = _reshape_to_daily_functions(df, value_col)

    results = []

    for station_id, g in daily.groupby("station_id"):
        if g.empty:
            continue

        # Optional filtering to fix one variable
        if fixed_by and fixed_value is not None:
            g = g[g[fixed_by] == fixed_value]
            if g.empty:
                continue

        # Build per-group matrices
        groups = {}
        for label, sub in g.groupby(group_by):
            curves = []
            for row in sub.itertuples():
                arr = np.array(row.lambda_curve, dtype=float)
                if normalize:
                    total = np.nansum(arr)
                    if total > 0:
                        arr = arr / total
                curves.append(arr)
            if len(curves) > 1:
                curves = _align_and_interpolate_curves(curves)
                if len(curves) > 0:
                    groups[label] = curves

        if len(groups) < 2:
            continue

        # Pointwise ANOVA across groups
        group_labels = list(groups.keys())
        group_arrays = list(groups.values())

        F_vals, p_vals = [], []
        for t in range(group_arrays[0].shape[1]):
            samples = [arr[:, t] for arr in group_arrays]
            samples = [s[~np.isnan(s)] for s in samples if len(s[~np.isnan(s)]) > 1]
            if len(samples) < 2:
                F_vals.append(np.nan)
                p_vals.append(np.nan)
                continue
            if np.allclose([np.nanstd(s) for s in samples], 0):
                F_vals.append(np.nan)
                p_vals.append(np.nan)
                continue
            F, p = f_oneway(*samples)
            F_vals.append(F)
            p_vals.append(p)

        F_vals = np.array(F_vals)
        p_vals = np.array(p_vals)
        frac_significant = np.nanmean(p_vals < 0.05)
        mean_p = np.nanmean(p_vals)

        results.append(
            {
                "station_id": station_id,
                "group_by": group_by,
                "fixed_by": fixed_by,
                "fixed_value": fixed_value,
                "n_groups": len(groups),
                "mean_p": mean_p,
                "frac_significant": frac_significant,
            }
        )

        if plot:
            title_suffix = (
                f" (only {fixed_by}={fixed_value})" if fixed_by and fixed_value else ""
            )
            plt.figure(figsize=(10, 4))
            for label, arr in zip(group_labels, group_arrays):
                mean_curve = np.nanmean(arr, axis=0)
                plt.plot(mean_curve, label=label)
            plt.title(
                f"Functional mean λ_{lambda_type} by {group_by} — Station {station_id}{title_suffix}"
            )
            plt.xlabel("15-min interval")
            plt.ylabel("Normalized λ(t)")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 3))
            plt.plot(p_vals, label="p-values over time", color="gray")
            plt.axhline(0.05, color="red", linestyle="--")
            plt.title(f"Pointwise ANOVA p-values — Station {station_id}{title_suffix}")
            plt.xlabel("15-min interval")
            plt.ylabel("p-value")
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)


stations = [9, 72, 103, 49]

# 1️⃣ Shape similarity within groups (e.g., date types)
corr_df = functional_correlation_analysis(
    stations, lambda_type="in", group_by="date_type", normalize=True, plot=True
)
print(corr_df)
