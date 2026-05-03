"""
realtime/adaptation.py

Three adaptive correction strategies for adjusting look-ahead forecasts
based on observed vs model discrepancy during a live session.

Option A — Multiplicative Scaling
    ratio = obs_count / model_count  (smoothed)
    future_intensity *= ratio

Option B — Bayesian Poisson-Gamma (default for all models)
    Prior:   λ_k ~ Gamma(α_k, β_k)   initialized from model mean/variance
    Update:  λ_k | N_obs ~ Gamma(α_k + N_obs, β_k + Δt)
    Ratio:   r_k = posterior_mean / prior_mean

Option C — Hawkes κ re-weight (Hawkes only)
    Re-estimates kappa from the recent observation window keeping α,β fixed.
    Uses an approximate closed-form MLE score update (single Newton step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SMOOTH_SIGMA = 1.5   # Gaussian kernel std in bins for spatial smoothing
_MIN_ALPHA    = 1e-3  # Minimum Gamma shape (numerical safety)
_MIN_BETA     = 1e-3  # Minimum Gamma rate  (numerical safety)


def _gaussian_smooth(arr: np.ndarray, sigma: float = _SMOOTH_SIGMA) -> np.ndarray:
    """Apply a 1-D Gaussian kernel to smooth an array of ratios/corrections."""
    K = len(arr)
    if K == 0 or sigma <= 0:
        return arr.copy()
    kernel_radius = max(1, int(3 * sigma))
    x = np.arange(-kernel_radius, kernel_radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


# ---------------------------------------------------------------------------
# Option B — Bayesian Poisson-Gamma
# ---------------------------------------------------------------------------

@dataclass
class BayesianAdaptation:
    """
    Poisson-Gamma conjugate intensity update per time bin.

    Parameters
    ----------
    prior_means : array of shape (K,)
        Model's predicted mean count per bin (λ_hat_k).
    prior_vars : array of shape (K,) | None
        Model's variance per bin.  If None, assumes Poisson (var = mean).
    dt_bin : float
        Duration of each bin in seconds (default 900 for 15-min).
    """

    prior_means: np.ndarray
    prior_vars:  Optional[np.ndarray] = None
    dt_bin:      float = 900.0

    # Posterior state (updated on each call to update())
    alpha_post: np.ndarray = field(init=False)
    beta_post:  np.ndarray = field(init=False)
    alpha_prior: np.ndarray = field(init=False)
    beta_prior:  np.ndarray = field(init=False)

    def __post_init__(self):
        mu = np.maximum(self.prior_means, _MIN_ALPHA)
        if self.prior_vars is None:
            var = mu   # Poisson assumption
        else:
            var = np.maximum(self.prior_vars, _MIN_ALPHA)

        # Method of moments: α = μ²/σ², β = μ/σ²
        self.alpha_prior = np.maximum(mu ** 2 / var, _MIN_ALPHA)
        self.beta_prior  = np.maximum(mu     / var, _MIN_BETA)

        # Initialise posterior to prior
        self.alpha_post = self.alpha_prior.copy()
        self.beta_post  = self.beta_prior.copy()

    def update(self, bin_indices: np.ndarray, observed_counts: np.ndarray) -> None:
        """
        Update posterior for the bins that were observed.

        Parameters
        ----------
        bin_indices : int array of which bins were observed (0-indexed)
        observed_counts : count in each corresponding bin
        """
        for idx, n_obs in zip(bin_indices, observed_counts):
            if 0 <= idx < len(self.alpha_post):
                self.alpha_post[idx] += float(n_obs)
                self.beta_post[idx]  += self.dt_bin

    def adjustment_ratios(self, smooth: bool = True) -> np.ndarray:
        """
        Return element-wise ratio: posterior_mean / prior_mean.
        Values > 1 → observed demand exceeds model; < 1 → under-demand.
        """
        post_mean  = self.alpha_post / self.beta_post
        prior_mean = self.alpha_prior / self.beta_prior
        ratios = post_mean / np.maximum(prior_mean, 1e-6)
        if smooth:
            ratios = _gaussian_smooth(ratios)
        return ratios

    def corrected_forecast(self, model_forecast: np.ndarray, bin_start: int) -> np.ndarray:
        """
        Apply adjustment ratios to a model forecast slice.
        model_forecast : array of length M (look-ahead bins)
        bin_start : index of first look-ahead bin in the full day profile.
        """
        ratios = self.adjustment_ratios()
        K = len(ratios)
        result = model_forecast.copy().astype(float)
        for i, val in enumerate(model_forecast):
            k = bin_start + i
            if 0 <= k < K:
                result[i] = val * ratios[k]
        return result

    def to_dict(self) -> dict:
        post_mean  = self.alpha_post / self.beta_post
        prior_mean = self.alpha_prior / self.beta_prior
        ratios = self.adjustment_ratios()
        return {
            "method":      "bayesian_gamma",
            "prior_means": prior_mean.tolist(),
            "post_means":  post_mean.tolist(),
            "ratios":      ratios.tolist(),
        }


# ---------------------------------------------------------------------------
# Option A — Multiplicative Scaling
# ---------------------------------------------------------------------------

@dataclass
class MultiplicativeAdaptation:
    """
    Simple ratio-based correction:  r = mean(obs_last_W) / mean(model_last_W)
    Exponential moving average over observation windows to reduce noise.
    """

    prior_means: np.ndarray
    ema_alpha:   float = 0.5   # EMA weight for new observations (0 = ignore new, 1 = forget all)

    _ratios: np.ndarray = field(init=False)

    def __post_init__(self):
        self._ratios = np.ones(len(self.prior_means))

    def update(self, bin_indices: np.ndarray, observed_counts: np.ndarray) -> None:
        K = len(self.prior_means)
        for idx, n_obs in zip(bin_indices, observed_counts):
            if 0 <= idx < K:
                model_val = max(self.prior_means[idx], 1e-3)
                new_ratio = float(n_obs) / model_val
                # Clip ratio to [0.1, 10] to prevent extreme corrections
                new_ratio = float(np.clip(new_ratio, 0.1, 10.0))
                self._ratios[idx] = (
                    (1 - self.ema_alpha) * self._ratios[idx] + self.ema_alpha * new_ratio
                )

    def adjustment_ratios(self, smooth: bool = True) -> np.ndarray:
        r = self._ratios.copy()
        if smooth:
            r = _gaussian_smooth(r)
        return r

    def corrected_forecast(self, model_forecast: np.ndarray, bin_start: int) -> np.ndarray:
        ratios = self.adjustment_ratios()
        K = len(ratios)
        result = model_forecast.copy().astype(float)
        for i, val in enumerate(model_forecast):
            k = bin_start + i
            if 0 <= k < K:
                result[i] = val * ratios[k]
        return result

    def to_dict(self) -> dict:
        return {
            "method": "multiplicative",
            "ratios": self.adjustment_ratios().tolist(),
        }


# ---------------------------------------------------------------------------
# Option C — Hawkes κ re-weight (single Newton step)
# ---------------------------------------------------------------------------

@dataclass
class HawkesKappaAdaptation:
    """
    Online re-estimation of the background scale kappa, keeping alpha/beta fixed.

    The MLE score for kappa given events in [t0, t0+W] is:
        dL/dκ = Σ_i μ_base(t_i) / λ(t_i) − ∫ μ_base dt

    A single Newton step from the current estimate gives an analytical update.
    This is much cheaper than a full refit.
    """

    kappa_init:  float
    alpha:       float
    beta:        float
    mu_blocks:   np.ndarray   # normalized background rates per bin (1/sec)
    dt_sec:      float = 900.0

    _kappa: float = field(init=False)

    def __post_init__(self):
        self._kappa = self.kappa_init

    @property
    def kappa(self) -> float:
        return self._kappa

    def update(
        self,
        t_events: np.ndarray,   # seconds from window start, sorted
        bin_start_sec: float,
        window_sec: float,
    ) -> None:
        """
        Update kappa using events in the observation window [bin_start_sec, bin_start_sec+window_sec].
        """
        if len(t_events) == 0:
            # No events → pull kappa toward zero slightly
            self._kappa = max(self._kappa * 0.9, 1e-4)
            return

        from src.workflow.scripts.models.hawkes.core import compute_Ah

        t_local = t_events - bin_start_sec
        t_local = t_local[(t_local >= 0) & (t_local < window_sec)]

        if len(t_local) == 0:
            return

        # Background intensity at each event time
        bin_idx = np.clip((t_local / self.dt_sec).astype(int), 0, len(self.mu_blocks) - 1)
        mu_at_t = self.mu_blocks[bin_idx]

        # Excitation term A(i) at each event time
        A = compute_Ah(t_local, self.alpha, self.beta)
        lambda_at_t = np.maximum(self._kappa * mu_at_t + A, 1e-10)

        # Score: dL/dκ = Σ μ/λ - ∫μ dt
        score = np.sum(mu_at_t / lambda_at_t) - np.sum(mu_at_t) * self.dt_sec
        # Fisher information (negative second derivative): Σ (μ/λ)²
        fisher = np.sum((mu_at_t / lambda_at_t) ** 2) + 1e-6

        # Newton step
        new_kappa = self._kappa + score / fisher
        self._kappa = float(np.clip(new_kappa, 1e-4, self._kappa * 5))

    def adjustment_ratio(self) -> float:
        """Return ratio of updated kappa to original kappa."""
        return self._kappa / max(self.kappa_init, 1e-6)

    def adjustment_ratios(self, n_bins: int = 76) -> np.ndarray:
        """Broadcast scalar ratio to array for compatibility with other methods."""
        return np.full(n_bins, self.adjustment_ratio())

    def corrected_forecast(self, model_forecast: np.ndarray, bin_start: int) -> np.ndarray:
        return model_forecast * self.adjustment_ratio()

    def to_dict(self) -> dict:
        return {
            "method":      "hawkes_kappa",
            "kappa_init":  self.kappa_init,
            "kappa_curr":  self._kappa,
            "ratio":       self.adjustment_ratio(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AdaptationMethod = Literal["bayesian", "multiplicative", "hawkes_kappa"]


def make_adaptation(
    method: AdaptationMethod,
    prior_means: np.ndarray,
    prior_vars: Optional[np.ndarray] = None,
    dt_bin: float = 900.0,
    # Hawkes-specific
    kappa_init: float = 1.0,
    alpha: float = 0.1,
    beta: float  = 0.2,
    mu_blocks: Optional[np.ndarray] = None,
) -> BayesianAdaptation | MultiplicativeAdaptation | HawkesKappaAdaptation:
    if method == "bayesian":
        return BayesianAdaptation(
            prior_means=prior_means,
            prior_vars=prior_vars,
            dt_bin=dt_bin,
        )
    elif method == "multiplicative":
        return MultiplicativeAdaptation(prior_means=prior_means)
    elif method == "hawkes_kappa":
        if mu_blocks is None:
            raise ValueError("mu_blocks required for hawkes_kappa adaptation")
        return HawkesKappaAdaptation(
            kappa_init=kappa_init,
            alpha=alpha,
            beta=beta,
            mu_blocks=mu_blocks,
            dt_sec=dt_bin,
        )
    else:
        raise ValueError(f"Unknown adaptation method: {method}")
