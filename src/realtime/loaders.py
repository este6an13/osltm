"""
realtime/loaders.py

Thin loaders that pull only what the real-time engine needs:
  - Model parameters (Hawkes, LGCP prior/posterior, Avg Profile)
  - Real check-in events from daily CSV files (seconds from midnight)
  - Day type inference from the database (WD / SA / SU / HO)
  - Available dates / model status helpers for the API
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIME_MIN = 400
TIME_MAX = 2300
TIME_STEP = 15  # minutes

_START_SEC = (TIME_MIN // 100) * 3600 + (TIME_MIN % 100) * 60
_END_SEC   = (TIME_MAX // 100) * 3600 + (TIME_MAX % 100) * 60
_T_TOTAL   = _END_SEC - _START_SEC   # seconds in observation window
_DT_SEC    = TIME_STEP * 60          # 900 seconds

# ---------------------------------------------------------------------------
# Day-type helpers
# ---------------------------------------------------------------------------

def query_day_type(date_str: str) -> Optional[str]:
    """
    Query osltm_v2.db for the day_type (WD/SA/SU/HO) of *date_str* (YYYYMMDD).
    Returns None if the date is not found in the counts table.
    """
    from src.db.session_v2 import SessionLocal
    from src.repo.v2.counts_15min.models import Counts15Min

    year  = int(date_str[:4])
    month = int(date_str[4:6])
    day   = int(date_str[6:8])

    db = SessionLocal()
    try:
        row = (
            db.query(Counts15Min.date_type)
            .filter(
                Counts15Min.year  == year,
                Counts15Min.month == month,
                Counts15Min.day   == day,
            )
            .first()
        )
        return row[0] if row else None
    finally:
        db.close()


def available_dates_with_day_type() -> dict[str, str]:
    """
    Return a mapping {YYYYMMDD: day_type} for every date present in the DB.
    Used by the UI to populate a date picker with metadata.
    """
    from src.db.session_v2 import SessionLocal
    from src.repo.v2.counts_15min.models import Counts15Min

    db = SessionLocal()
    try:
        rows = (
            db.query(Counts15Min.year, Counts15Min.month, Counts15Min.day, Counts15Min.date_type)
            .distinct()
            .all()
        )
        result: dict[str, str] = {}
        for yr, mo, dy, dt in rows:
            key = f"{yr:04d}{mo:02d}{dy:02d}"
            result[key] = dt
        return result
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Real event loader
# ---------------------------------------------------------------------------

def _extract_station_code(estacion_str: str) -> str:
    """Extract zero-padded 5-digit station code from '(XXXXX) Name' strings."""
    m = re.search(r"\((\d+)\)", str(estacion_str))
    if m:
        return m.group(1).zfill(5)
    return str(estacion_str).strip().zfill(5)


def load_real_events(
    date_str: str,
    station_codes: list[str],
) -> dict[str, np.ndarray]:
    """
    Load real check-in events for *date_str* (YYYYMMDD) and the given stations.

    Returns
    -------
    dict mapping station_code -> sorted array of seconds-from-midnight,
    filtered to [TIME_MIN … TIME_MAX) window.
    Empty array if no data for that station.
    """
    csv_path = Path(f"data/check_ins/daily/{date_str}.csv")

    result: dict[str, np.ndarray] = {sc: np.array([]) for sc in station_codes}

    if not csv_path.exists():
        return result

    try:
        df = pd.read_csv(
            csv_path,
            usecols=["Fecha_Transaccion", "Estacion_Parada"],
            parse_dates=["Fecha_Transaccion"],
        )
    except Exception:
        try:
            df = pd.read_csv(
                csv_path,
                usecols=["Fecha_Transaccion", "Estacion"],
                parse_dates=["Fecha_Transaccion"],
            )
            df = df.rename(columns={"Estacion": "Estacion_Parada"})
        except Exception:
            return result

    df["station_code"] = df["Estacion_Parada"].apply(_extract_station_code)
    df = df[df["station_code"].isin(station_codes)].copy()

    if df.empty:
        return result

    dt = df["Fecha_Transaccion"]
    df["sec"] = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second

    # Filter to observation window
    df = df[(df["sec"] >= _START_SEC) & (df["sec"] <= _END_SEC)].copy()
    df["sec_shifted"] = df["sec"] - _START_SEC

    for sc in station_codes:
        sdf = df[df["station_code"] == sc]
        result[sc] = np.sort(sdf["sec_shifted"].values.astype(float))

    return result


# ---------------------------------------------------------------------------
# Availability helpers (for UI to show which models are ready)
# ---------------------------------------------------------------------------

def build_model_inventory(results_base: Path) -> dict[str, dict[str, list[str]]]:
    """
    Scans the results directory and returns a full inventory of fitted models:
    { station_code: { day_type: [model_key, ...] } }
    """
    inventory: dict[str, dict[str, list[str]]] = {}

    def _add(sc: str, dt: str, model: str) -> None:
        if sc not in inventory:
            inventory[sc] = {}
        if dt not in inventory[sc]:
            inventory[sc][dt] = []
        if model not in inventory[sc][dt]:
            inventory[sc][dt].append(model)

    # Hawkes
    for path in sorted(results_base.glob("hawkes_fit/**/hawkes_params_checkins.csv"), reverse=True):
        try:
            df = pd.read_csv(path, dtype={"station_code": str})
            for _, row in df.iterrows():
                _add(str(row["station_code"]).zfill(5), str(row["day_type"]), "hawkes")
        except Exception:
            pass

    # LGCP Prior
    for path in sorted(results_base.glob("lgcp_twostage/**/lgcp_kernel_params_checkins.csv"), reverse=True):
        try:
            df = pd.read_csv(path, dtype={"station_code": str})
            df = df[df["is_selected"] == True]
            for _, row in df.iterrows():
                _add(str(row["station_code"]).zfill(5), str(row["day_type"]), "lgcp_prior")
        except Exception:
            pass

    # LGCP Posterior
    for path in sorted(results_base.glob("lgcp_bayesian/**/lgcp_posterior_params_checkins.csv"), reverse=True):
        try:
            df = pd.read_csv(path, dtype={"station_code": str})
            for _, row in df.iterrows():
                _add(str(row["station_code"]).zfill(5), str(row["day_type"]), "lgcp_posterior")
        except Exception:
            pass

    # Avg Profile
    for path in sorted(results_base.glob("avg_profile/**/avg_profile_params_checkins.csv"), reverse=True):
        try:
            df = pd.read_csv(path, dtype={"station_code": str})
            for _, row in df.iterrows():
                _add(str(row["station_code"]).zfill(5), str(row["day_type"]), "avg_profile")
        except Exception:
            pass

    return inventory


def check_model_availability(
    results_base: Path,
    station_codes: list[str],
    day_type: str,
) -> dict[str, bool | str]:
    """
    Check which models have fitted parameters for the given station(s)/day_type.
    Returns a dict {model_key: True | "partial" | False}.
    Models: hawkes, lgcp_prior, lgcp_posterior, avg_profile
    """

    def _has_hawkes() -> bool:
        for path in sorted(results_base.glob("hawkes_fit/**/hawkes_params_checkins.csv"), reverse=True):
            try:
                df = pd.read_csv(path)
                df["station_code"] = df["station_code"].astype(str).str.zfill(5)
                has = all(
                    not df[(df["station_code"] == sc) & (df["day_type"] == day_type)].empty
                    for sc in station_codes
                )
                if has:
                    return True
            except Exception:
                continue
        return False

    def _has_lgcp_prior() -> bool:
        for path in sorted(results_base.glob("lgcp_twostage/**/lgcp_kernel_params_checkins.csv"), reverse=True):
            try:
                df = pd.read_csv(path)
                df["station_code"] = df["station_code"].astype(str).str.zfill(5)
                df = df[df["is_selected"] == True]
                has = all(
                    not df[(df["station_code"] == sc) & (df["day_type"] == day_type)].empty
                    for sc in station_codes
                )
                if has:
                    return True
            except Exception:
                continue
        return False

    def _has_lgcp_posterior() -> bool:
        for path in sorted(results_base.glob("lgcp_bayesian/**/lgcp_posterior_params_checkins.csv"), reverse=True):
            try:
                df = pd.read_csv(path)
                df["station_code"] = df["station_code"].astype(str).str.zfill(5)
                has = all(
                    not df[(df["station_code"] == sc) & (df["day_type"] == day_type)].empty
                    for sc in station_codes
                )
                if has:
                    return True
            except Exception:
                continue
        return False

    def _has_avg_profile() -> bool:
        for path in sorted(results_base.glob("avg_profile/**/avg_profile_params_checkins.csv"), reverse=True):
            try:
                df = pd.read_csv(path)
                df["station_code"] = df["station_code"].astype(str).str.zfill(5)
                has = all(
                    not df[(df["station_code"] == sc) & (df["day_type"] == day_type)].empty
                    for sc in station_codes
                )
                if has:
                    return True
            except Exception:
                continue
        return False

    return {
        "hawkes":         _has_hawkes(),
        "lgcp_prior":     _has_lgcp_prior(),
        "lgcp_posterior": _has_lgcp_posterior(),
        "avg_profile":    _has_avg_profile(),
    }


# ---------------------------------------------------------------------------
# Parameter loaders — one per model type
# ---------------------------------------------------------------------------

def load_hawkes_params(
    results_base: Path,
    station_codes: list[str],
    day_type: str,
) -> dict[str, dict]:
    """
    Load median Hawkes params (kappa, alpha, beta) per station for the given day_type.
    Searches the most recent experiment under hawkes_fit/.
    Returns {station_code: {kappa, alpha, beta, profile}} or raises if not found.
    """
    csv_candidates = sorted(
        results_base.glob("hawkes_fit/**/hawkes_params_checkins.csv"), reverse=True
    )
    if not csv_candidates:
        raise FileNotFoundError("No hawkes_params_checkins.csv found under results/hawkes_fit/")

    fit_df: pd.DataFrame | None = None
    for path in csv_candidates:
        try:
            df = pd.read_csv(path)
            df["station_code"] = df["station_code"].astype(str).str.zfill(5)
            df = df[df["day_type"] == day_type]
            df = df[df["station_code"].isin(station_codes)]
            if not df.empty:
                fit_df = df
                break
        except Exception:
            continue

    if fit_df is None or fit_df.empty:
        raise ValueError(
            f"No Hawkes params found for stations={station_codes}, day_type={day_type}"
        )

    out: dict[str, dict] = {}
    for sc in station_codes:
        sdf = fit_df[fit_df["station_code"] == sc]
        if sdf.empty:
            raise ValueError(f"No Hawkes params for station {sc}, day_type={day_type}")
        out[sc] = {
            "kappa": float(sdf["kappa"].median()),
            "alpha": float(sdf["alpha"].median()),
            "beta":  float(sdf["beta"].median()),
        }
    return out


def load_lgcp_prior_params(
    results_base: Path,
    station_codes: list[str],
    day_type: str,
) -> dict[str, dict]:
    """
    Load LGCP prior params (mu, L_chol) per station for the given day_type.
    Returns {station_code: {kernel, sigma2, ell, mu, time_hours}}.
    """
    csv_candidates = sorted(
        results_base.glob("lgcp_twostage/**/lgcp_kernel_params_checkins.csv"), reverse=True
    )
    if not csv_candidates:
        raise FileNotFoundError("No lgcp_kernel_params_checkins.csv found.")

    kernel_df: pd.DataFrame | None = None
    for path in csv_candidates:
        try:
            df = pd.read_csv(path)
            df["station_code"] = df["station_code"].astype(str).str.zfill(5)
            df = df[(df["is_selected"] == True) & (df["day_type"] == day_type)]
            df = df[df["station_code"].isin(station_codes)]
            if not df.empty:
                kernel_df = df
                break
        except Exception:
            continue

    if kernel_df is None or kernel_df.empty:
        raise ValueError(f"No LGCP prior params for stations={station_codes}, day_type={day_type}")

    # We also need training data to build mu
    from src.workflow.data_loader import load_data
    data = load_data(
        station_codes=station_codes,
        include_checkins=True,
        include_checkouts=False,
        time_min=TIME_MIN,
        time_max=TIME_MAX,
        time_step=TIME_STEP,
    )
    df_all = data["checkins"]
    df_all["station_code"] = df_all["station_code"].astype(str).str.zfill(5)

    time_cols = sorted([c for c in df_all.columns if c.startswith("t_")],
                       key=lambda c: int(c.replace("t_", "")))
    time_hours = np.array([
        int(c.replace("t_", "")) // 100 + (int(c.replace("t_", "")) % 100) / 60.0
        for c in time_cols
    ])

    out: dict[str, dict] = {}
    for sc in station_codes:
        krow = kernel_df[kernel_df["station_code"] == sc]
        if krow.empty:
            raise ValueError(f"No LGCP prior kernel for station {sc}, day_type={day_type}")
        krow = krow.iloc[0]

        sdf = df_all[(df_all["station_code"] == sc) & (df_all["date_type"] == day_type)]
        if sdf.empty:
            raise ValueError(f"No training data for station {sc}, day_type={day_type}")

        counts = sdf[time_cols].fillna(0).values.astype(float)
        mu = np.log(counts.mean(axis=0) + 0.5)

        out[sc] = {
            "kernel":     krow["kernel"],
            "sigma2":     float(krow["sigma2"]),
            "ell":        float(krow["ell_hours"]),
            "mu":         mu,
            "time_hours": time_hours,
            "time_cols":  time_cols,
        }
    return out


def load_lgcp_posterior_params(
    results_base: Path,
    station_codes: list[str],
    day_type: str,
) -> dict[str, dict]:
    """
    Load LGCP posterior params (z_map, H, kernel) per station for the given day_type.
    Requires that load_lgcp_prior_params also succeeds (shares kernel file).
    """
    prior_params = load_lgcp_prior_params(results_base, station_codes, day_type)

    csv_candidates = sorted(
        results_base.glob("lgcp_bayesian/**/lgcp_posterior_params_checkins.csv"), reverse=True
    )
    if not csv_candidates:
        raise FileNotFoundError("No lgcp_posterior_params_checkins.csv found.")

    posterior_df: pd.DataFrame | None = None
    for path in csv_candidates:
        try:
            df = pd.read_csv(path)
            df["station_code"] = df["station_code"].astype(str).str.zfill(5)
            df = df[(df["day_type"] == day_type) & (df["station_code"].isin(station_codes))]
            if not df.empty:
                posterior_df = df
                break
        except Exception:
            continue

    if posterior_df is None or posterior_df.empty:
        raise ValueError(f"No LGCP posterior params for stations={station_codes}, day_type={day_type}")

    out: dict[str, dict] = {}
    for sc in station_codes:
        sp = posterior_df[posterior_df["station_code"] == sc].copy()
        if sp.empty:
            raise ValueError(f"No posterior for station {sc}, day_type={day_type}")

        sp["_t_int"] = sp["time_bin"].str.replace("t_", "", regex=False).astype(int)
        sp = sp.sort_values("_t_int").reset_index(drop=True)
        z_map = sp["z_posterior_mean"].values.astype(float)

        pp = prior_params[sc]
        out[sc] = {**pp, "z_map": z_map}
    return out


def load_avg_profile_params(
    results_base: Path,
    station_codes: list[str],
    day_type: str,
) -> dict[str, dict]:
    """
    Load average profile params (mean/std per bin) for the given stations/day_type.
    """
    csv_candidates = sorted(
        results_base.glob("avg_profile/**/avg_profile_params_checkins.csv"), reverse=True
    )
    if not csv_candidates:
        raise FileNotFoundError("No avg_profile_params_checkins.csv found.")

    profile_df: pd.DataFrame | None = None
    for path in csv_candidates:
        try:
            df = pd.read_csv(path)
            df["station_code"] = df["station_code"].astype(str).str.zfill(5)
            df = df[(df["day_type"] == day_type) & (df["station_code"].isin(station_codes))]
            if not df.empty:
                profile_df = df
                break
        except Exception:
            continue

    if profile_df is None or profile_df.empty:
        raise ValueError(f"No Avg Profile params for stations={station_codes}, day_type={day_type}")

    # Reconstruct time cols from column names (e.g. "t_400_mean")
    mean_cols = sorted(
        [c for c in profile_df.columns if c.endswith("_mean") and c.startswith("t_")],
        key=lambda c: int(c.replace("t_", "").replace("_mean", "")),
    )
    time_cols = [c.replace("_mean", "") for c in mean_cols]
    time_hours = np.array([
        int(c.replace("t_", "")) // 100 + (int(c.replace("t_", "")) % 100) / 60.0
        for c in time_cols
    ])

    out: dict[str, dict] = {}
    for sc in station_codes:
        row = profile_df[profile_df["station_code"] == sc]
        if row.empty:
            raise ValueError(f"No Avg Profile for station {sc}, day_type={day_type}")
        row = row.iloc[0]
        means = np.array([float(row[c]) for c in mean_cols])
        std_cols = [c.replace("_mean", "_std") for c in mean_cols]
        stds = np.array([
            float(row[c]) if c in row.index else 0.0 for c in std_cols
        ])
        out[sc] = {
            "means":      means,
            "stds":       stds,
            "time_hours": time_hours,
            "time_cols":  time_cols,
        }
    return out
