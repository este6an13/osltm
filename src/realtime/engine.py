"""
realtime/engine.py

Stateful simulation engine for one real-time session.

Responsibilities
----------------
1. Pre-generate ALL model events (seconds from window start) at session creation.
2. Load real check-in events from the daily CSV (if available).
3. Provide fast tick-driven event slicing (get_events_in_window).
4. Manage adaptive correction state per station.
5. Produce look-ahead forecasts (raw + corrected).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from src.realtime.loaders import (
    TIME_MIN, TIME_MAX, TIME_STEP,
    _START_SEC, _END_SEC, _T_TOTAL, _DT_SEC,
    load_real_events,
    load_hawkes_params,
    load_lgcp_prior_params,
    load_lgcp_posterior_params,
    load_avg_profile_params,
    load_cluster_params,
    query_day_type,
)
from src.realtime.adaptation import (
    AdaptationMethod,
    make_adaptation,
    BayesianAdaptation,
    MultiplicativeAdaptation,
    HawkesKappaAdaptation,
)

ModelType = Literal["hawkes", "lgcp_prior", "lgcp_posterior", "avg_profile", "cluster"]

RESULTS_BASE = Path("src/workflow/results")

# How many 15-min bins in the observation window
_N_BINS = int((_T_TOTAL) / _DT_SEC)

# Time-hours array for the full day profile
_TIME_HOURS = np.array([
    (TIME_MIN // 100 + (TIME_MIN % 100) / 60.0) + i * (TIME_STEP / 60.0)
    for i in range(_N_BINS)
])

# ---------------------------------------------------------------------------
# Internal simulation helpers
# ---------------------------------------------------------------------------

def _kernel_se(t: np.ndarray, sigma2: float, ell: float) -> np.ndarray:
    diffs = t[:, None] - t[None, :]
    return sigma2 * np.exp(-0.5 * (diffs / ell) ** 2)

def _kernel_matern32(t: np.ndarray, sigma2: float, ell: float) -> np.ndarray:
    diffs = np.abs(t[:, None] - t[None, :])
    r = np.sqrt(3) * diffs / ell
    return sigma2 * (1 + r) * np.exp(-r)


def _simulate_lgcp_prior(params: dict, rng: np.random.Generator) -> np.ndarray:
    """Draw one LGCP prior day → returns sorted seconds from window start."""
    mu         = params["mu"]
    time_hours = params["time_hours"]
    sigma2     = params["sigma2"]
    ell        = params["ell"]
    kernel_fn  = _kernel_se if params["kernel"] == "SE" else _kernel_matern32

    C = kernel_fn(time_hours, sigma2, ell)
    C += 1e-6 * np.eye(len(time_hours))
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        C += 1e-4 * np.eye(len(time_hours))
        L = np.linalg.cholesky(C)

    z   = mu + L @ rng.standard_normal(len(mu))
    lam = np.exp(z)
    counts = rng.poisson(lam).astype(int)

    events = []
    for k, n in enumerate(counts):
        if n > 0:
            t_start = k * _DT_SEC
            t_end   = min((k + 1) * _DT_SEC, _T_TOTAL)
            events.append(rng.uniform(t_start, t_end, size=n))
    return np.sort(np.concatenate(events)) if events else np.array([])


def _simulate_lgcp_posterior(params: dict, rng: np.random.Generator) -> np.ndarray:
    """Draw one LGCP posterior day → returns sorted seconds from window start."""
    z_map      = params["z_map"]
    time_hours = params["time_hours"]
    sigma2     = params["sigma2"]
    ell        = params["ell"]
    eta2       = params.get("eta2", 0.0)
    kernel_fn  = _kernel_se if params["kernel"] == "SE" else _kernel_matern32

    K  = len(time_hours)
    C  = kernel_fn(time_hours, sigma2, ell)
    C  += eta2 * np.eye(K) + 1e-6 * np.eye(K)
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C)

    W = np.diag(np.exp(z_map))
    H = W + C_inv + 1e-6 * np.eye(K)
    try:
        L_H = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        H += 1e-4 * np.eye(K)
        L_H = np.linalg.cholesky(H)

    eps = rng.standard_normal(K)
    z   = z_map + np.linalg.solve(L_H.T, eps)
    lam = np.exp(z)
    counts = rng.poisson(lam).astype(int)

    events = []
    for k, n in enumerate(counts):
        if n > 0:
            t_start = k * _DT_SEC
            t_end   = min((k + 1) * _DT_SEC, _T_TOTAL)
            events.append(rng.uniform(t_start, t_end, size=n))
    return np.sort(np.concatenate(events)) if events else np.array([])


def _simulate_hawkes_day(params: dict, rng: np.random.Generator) -> np.ndarray:
    from src.workflow.scripts.models.hawkes.core import simulate_hawkes_branching
    profile = {
        "mu_blocks": params["mu_blocks"],
        "dt_sec":    _DT_SEC,
        "T_total":   _T_TOTAL,
    }
    return simulate_hawkes_branching(
        [params["kappa"], params["alpha"], params["beta"]], profile, rng
    )


def _simulate_avg_profile_day(params: dict, rng: np.random.Generator) -> np.ndarray:
    means   = np.maximum(params["means"], 0)
    counts  = rng.poisson(means).astype(int)
    events  = []
    for k, n in enumerate(counts):
        if n > 0:
            t_start = k * _DT_SEC
            t_end   = min((k + 1) * _DT_SEC, _T_TOTAL)
            events.append(rng.uniform(t_start, t_end, size=n))
    return np.sort(np.concatenate(events)) if events else np.array([])


def _simulate_cluster_day(params: dict, rng: np.random.Generator) -> np.ndarray:
    from src.workflow.scripts.models.cluster.core import simulate_cluster_process
    p = {**params, "dt_sec": _DT_SEC, "T_total": _T_TOTAL}
    return simulate_cluster_process(p, rng)


def _build_hawkes_mu_blocks(params: dict) -> np.ndarray:
    """
    Build normalized background rate blocks for Hawkes adaptation (Option C).
    Uses a flat uniform profile if raw counts are not available.
    """
    # params["mu_blocks"] may already be stored; if not build uniform
    return params.get("mu_blocks", np.full(_N_BINS, 1.0 / (_N_BINS * _DT_SEC)))


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class RealtimeSession:
    session_id:    str
    date_str:      str          # YYYYMMDD
    day_type:      str          # WD / SA / SU / HO
    model:         ModelType
    station_codes: list[str]
    count_type:    str          # checkins / checkouts
    clock_start_sec: float      # seconds from window start (0 = TIME_MIN)
    speed:         float        # 1.0 = real time

    adaptation_method: AdaptationMethod = "bayesian"

    # Pre-generated events: station → sorted seconds from window start
    model_events:  dict[str, np.ndarray] = field(default_factory=dict)
    real_events:   dict[str, np.ndarray] = field(default_factory=dict)
    has_real_data: bool = False

    # Station display name map (code → name)
    station_names: dict[str, str] = field(default_factory=dict)

    # Model prior means per bin (used for adaptation init)
    _prior_means:  dict[str, np.ndarray] = field(default_factory=dict)
    _prior_vars:   dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    _hawkes_params: dict[str, dict] = field(default_factory=dict)

    # Adaptation state per station
    _adaptation:   dict[str, object] = field(default_factory=dict)

    # Look-ahead horizon
    lookahead_sec: float = 3600.0   # 1 hour

    def get_events_in_window(
        self, t_start: float, t_end: float
    ) -> dict[str, dict[str, list[float]]]:
        """
        Return model and real events in [t_start, t_end) seconds from window start.
        Also triggers adaptation update if a full bin was just completed.
        """
        result: dict[str, dict[str, list[float]]] = {}

        for sc in self.station_codes:
            m_ev = self.model_events.get(sc, np.array([]))
            r_ev = self.real_events.get(sc, np.array([]))

            mask_m = (m_ev >= t_start) & (m_ev < t_end)
            mask_r = (r_ev >= t_start) & (r_ev < t_end)

            result[sc] = {
                "model": m_ev[mask_m].tolist(),
                "real":  r_ev[mask_r].tolist(),
            }

        # Adaptation update: check which bins just closed
        self._maybe_update_adaptation(t_end)

        return result

    def _maybe_update_adaptation(self, t_now: float) -> None:
        """Check if any 15-min bins have fully elapsed and update adaptation state."""
        # Closed bins: those where bin_end <= t_now
        n_closed = int(t_now / _DT_SEC)

        for sc in self.station_codes:
            adapt = self._adaptation.get(sc)
            if adapt is None:
                continue
            adapt_state = adapt.get("state")
            last_updated_bin = adapt.get("last_bin", -1)
            if adapt_state is None:
                continue

            for b in range(last_updated_bin + 1, n_closed):
                b_start = b * _DT_SEC
                b_end   = (b + 1) * _DT_SEC

                # Count real events in this bin
                r_ev = self.real_events.get(sc, np.array([]))
                n_obs = int(np.sum((r_ev >= b_start) & (r_ev < b_end)))

                if isinstance(adapt_state, HawkesKappaAdaptation):
                    # Option C: feed events array to kappa updater
                    r_in_bin = r_ev[(r_ev >= b_start) & (r_ev < b_end)]
                    adapt_state.update(r_in_bin, b_start, _DT_SEC)
                else:
                    adapt_state.update(np.array([b]), np.array([n_obs]))

                adapt["last_bin"] = b

    def get_forecast(
        self, t_now: float
    ) -> dict[str, dict[str, list[float]]]:
        """
        Return binned look-ahead forecast (raw + corrected) per station.
        Covers [t_now, t_now + lookahead_sec] discretized into 15-min bins.
        """
        bin_start = int(t_now / _DT_SEC)
        bin_end   = min(_N_BINS, bin_start + int(self.lookahead_sec / _DT_SEC) + 1)

        result: dict[str, dict[str, list[float]]] = {}

        for sc in self.station_codes:
            prior = self._prior_means.get(sc, np.zeros(_N_BINS))
            model_slice = prior[bin_start:bin_end]

            adapt = self._adaptation.get(sc, {})
            adapt_state = adapt.get("state") if adapt else None

            if adapt_state is not None:
                corrected = adapt_state.corrected_forecast(model_slice, bin_start)
            else:
                corrected = model_slice.copy()

            # Time axis (hours) for the forecast bins
            t_hours = _TIME_HOURS[bin_start:bin_end].tolist()

            result[sc] = {
                "time_hours":  t_hours,
                "model_raw":   model_slice.tolist(),
                "corrected":   corrected.tolist(),
            }

        return result

    def get_adaptation_state(self) -> dict[str, dict]:
        """Return serializable adaptation state per station."""
        out: dict[str, dict] = {}
        for sc in self.station_codes:
            adapt = self._adaptation.get(sc, {})
            state = adapt.get("state")
            out[sc] = state.to_dict() if state is not None else {}
        return out


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def create_session(
    station_codes:     list[str],
    model:             ModelType,
    day_type:          str,
    count_type:        str = "checkins",
    date_str:          str = "",
    adaptation_method: AdaptationMethod = "bayesian",
    clock_start_hhmm:  int = TIME_MIN,      # e.g. 700 for 07:00
    speed:             float = 1.0,
    lookahead_min:     float = 60.0,
    seed:              int = 42,
) -> RealtimeSession:
    """
    Create and fully initialize a RealtimeSession:
    """
    station_codes = [sc.zfill(5) for sc in station_codes]

    # Convert clock_start_hhmm to seconds from window start
    h = clock_start_hhmm // 100
    m = clock_start_hhmm % 100
    clock_start_absolute = h * 3600 + m * 60
    clock_start_sec = max(0.0, float(clock_start_absolute - _START_SEC))

    session = RealtimeSession(
        session_id        = str(uuid.uuid4()),
        date_str          = date_str,
        day_type          = day_type,
        model             = model,
        station_codes     = station_codes,
        count_type        = count_type,
        clock_start_sec   = clock_start_sec,
        speed             = speed,
        adaptation_method = adaptation_method,
        lookahead_sec     = lookahead_min * 60.0,
    )

    # 2. Load model params + generate events
    rng = np.random.default_rng(seed)
    _generate_model_events(session, rng)

    # 3. Real events
    real = load_real_events(date_str, station_codes, count_type)
    session.real_events   = real
    session.has_real_data = any(len(v) > 0 for v in real.values())

    # 4. Adaptation
    _init_adaptation(session)

    return session


def _generate_model_events(session: RealtimeSession, rng: np.random.Generator) -> None:
    model = session.model
    scs   = session.station_codes
    dt    = session.day_type
    ct    = session.count_type

    if model == "hawkes":
        params_map = load_hawkes_params(RESULTS_BASE, scs, dt, ct)
        # Build mu_blocks for each station from a uniform profile (conservative)
        for sc, p in params_map.items():
            p["mu_blocks"] = np.full(_N_BINS, 1.0 / (_N_BINS * _DT_SEC))
        for sc, p in params_map.items():
            session.model_events[sc]  = _simulate_hawkes_day(p, rng)
            session._prior_means[sc]  = p["kappa"] * p["mu_blocks"] * _DT_SEC
            session._hawkes_params[sc] = p

    elif model == "lgcp_prior":
        params_map = load_lgcp_prior_params(RESULTS_BASE, scs, dt, ct)
        for sc, p in params_map.items():
            session.model_events[sc] = _simulate_lgcp_prior(p, rng)
            session._prior_means[sc] = np.exp(p["mu"])
            session._prior_vars[sc]  = np.exp(p["mu"])   # Poisson approx

    elif model == "lgcp_posterior":
        params_map = load_lgcp_posterior_params(RESULTS_BASE, scs, dt, ct)
        for sc, p in params_map.items():
            session.model_events[sc] = _simulate_lgcp_posterior(p, rng)
            session._prior_means[sc] = np.exp(p["z_map"])
            session._prior_vars[sc]  = np.exp(p["z_map"])

    elif model == "avg_profile":
        params_map = load_avg_profile_params(RESULTS_BASE, scs, dt, ct)
        for sc, p in params_map.items():
            session.model_events[sc] = _simulate_avg_profile_day(p, rng)
            session._prior_means[sc] = p["means"]
            session._prior_vars[sc]  = p["stds"] ** 2

    elif model.startswith("cluster"):
        method_parts = model.split("_", 1)
        cluster_method = method_parts[1] if len(method_parts) > 1 else None
        
        params_map = load_cluster_params(RESULTS_BASE, scs, dt, ct, method=cluster_method)
        for sc, p in params_map.items():
            session.model_events[sc] = _simulate_cluster_day(p, rng)
            # Expected events per bin = background + parents * children_per_parent
            expected_noise = p["noise_mu_blocks"] * _DT_SEC
            expected_children = p["centroid_mu_blocks"] * _DT_SEC * p["cluster_size_mean"]
            session._prior_means[sc] = expected_noise + expected_children
            session._prior_vars[sc] = session._prior_means[sc]

    else:
        raise ValueError(f"Unknown model: {model}")


def _init_adaptation(session: RealtimeSession) -> None:
    method = session.adaptation_method
    for sc in session.station_codes:
        prior_means = session._prior_means.get(sc, np.ones(_N_BINS))
        prior_vars  = session._prior_vars.get(sc)

        if method == "hawkes_kappa" and session.model == "hawkes":
            hp = session._hawkes_params.get(sc, {})
            state = make_adaptation(
                method      = "hawkes_kappa",
                prior_means = prior_means,
                kappa_init  = hp.get("kappa", 1.0),
                alpha       = hp.get("alpha", 0.1),
                beta        = hp.get("beta", 0.2),
                mu_blocks   = hp.get("mu_blocks", np.full(_N_BINS, 1.0 / (_N_BINS * _DT_SEC))),
                dt_bin      = _DT_SEC,
            )
        elif method == "multiplicative":
            state = make_adaptation("multiplicative", prior_means, dt_bin=_DT_SEC)
        elif method == "trend":
            state = make_adaptation("trend", prior_means, dt_bin=_DT_SEC)
        else:
            # Default: bayesian
            state = make_adaptation("bayesian", prior_means, prior_vars, dt_bin=_DT_SEC)

        session._adaptation[sc] = {"state": state, "last_bin": -1}
