import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kstest, nbinom
from sklearn.neighbors import KernelDensity
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------- Reproducibility ----------
np.random.seed(0)  # set before generating randomized PIT


# ---------- Method-of-moments dispersion ----------
def kmom_from_counts(counts, means):
    """Compute MoM estimate of k from per-minute empirical mean/var.
    counts: array of observed counts per minute (may include repeated days)
    means: array of predicted means (lambda_t) per minute
    Both arrays must align length-wise.
    We compute empirical var across minutes by grouping by minute index
    if you have multiple days; with single day this is noisy.
    """
    # For single-day aggregated counts, MoM is noisy; still:
    _ = (counts - counts.mean()) ** 2  # if only one sample per minute this is wrong
    # Instead if counts is series indexed by minute and repeated over many days, compute var per minute:
    # Here we assume 'counts' is per-minute counts for a single day; better DO per-minute across many days.
    # We'll compute a global method-of-moments k:
    mean_overall = counts.mean()
    var_overall = counts.var(ddof=1)
    k_mom = max((var_overall - mean_overall) / (mean_overall**2), 0.0)
    return k_mom


# ---------- Mean-Variance scatter + NB curve ----------
def plot_mean_vs_var(minute_grid, intensity, count_per_min, k):
    # If you have only a single day, aggregate into larger bins to stabilize variance:
    # e.g., 5-minute bins
    bin_size = 5
    mg = minute_grid.flatten()
    idx_bins = (mg - mg.min()) // bin_size
    df_counts = pd.DataFrame(
        {
            "bin": idx_bins,
            "minute": mg,
            "mean_lambda": intensity,
            "count": count_per_min.values,
        }
    )
    grouped = (
        df_counts.groupby("bin")
        .agg(
            {
                "mean_lambda": "mean",
                "count": "mean",  # note: for a single day this is same as raw; with multiple days use var etc.
            }
        )
        .reset_index()
    )

    # empirical mean and var per bin (if you have multiple days, compute var across days)
    # For single-day demonstration we'll just compare mean vs count (no var).
    plt.figure(figsize=(7, 5))
    plt.scatter(grouped["mean_lambda"], grouped["count"], alpha=0.6, label="Empirical")
    # NB variance curve (using estimated k)
    k_est = k  # from your fitted k variable
    mu_grid = np.linspace(0.01, max(grouped["mean_lambda"].max(), 1.0), 200)
    var_nb = mu_grid + k_est * mu_grid**2
    plt.plot(mu_grid, var_nb, linestyle="--", label=f"NB Var (k={k_est:.4f})")
    plt.plot(mu_grid, mu_grid, linestyle=":", label="Poisson Var=Mean")
    plt.xlabel("Mean (μ)")
    plt.ylabel("Empirical count / Var")
    plt.title("Mean vs Empirical Count (binned)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------- Bootstrap CI for k (simple) ----------
def bootstrap_k(intensity, counts, n_boot=200):
    """Bootstrap samples of per-minute counts to get CI for k.
    This resamples minutes with replacement — conservative but quick.
    """
    ks = []
    n = len(counts)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        counts_b = counts[idx]
        _ = intensity[idx]
        # method-of-moments for bootstrap sample
        mean_b = counts_b.mean()
        var_b = counts_b.var(ddof=1)
        k_b = max((var_b - mean_b) / (mean_b**2) if mean_b > 0 else 0.0, 0.0)
        ks.append(k_b)
    ks = np.array(ks)
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return ks.mean(), lo, hi


def fit_nb_dispersion(mean_vals, count_vals):
    """Fit NB dispersion parameter k via MLE."""
    # Avoid zeros
    mean_vals = np.maximum(mean_vals, 1e-6)

    def neg_log_likelihood(log_k):
        k = np.exp(log_k)
        var = mean_vals + k * mean_vals**2
        p = mean_vals / var
        r = mean_vals**2 / (var - mean_vals)
        ll = nbinom.logpmf(count_vals, r, p)
        return -np.sum(ll)

    res = minimize(neg_log_likelihood, x0=np.log(0.1))
    return np.exp(res.x[0])


def estimate_intensity_kde(df: pd.DataFrame, station_code: str, bandwidth=5):
    """
    Estimate intensity of arrivals using KDE.
    Returns:
        - minute_grid (1D array)
        - intensity (1D array, λ(t))
        - count_per_min (real observed counts)
    """

    if df.empty:
        print("⚠️ No data found for this station.")
        return None, None, None

    # --- SAFE COPY ---
    df = df.copy()

    # --- Timestamp handling ---
    df.loc[:, "timestamp"] = pd.to_datetime(df["Fecha_Transaccion"])
    df.loc[:, "time_int"] = df["timestamp"].dt.hour * 100 + df["timestamp"].dt.minute

    # Operating window 04:00–23:00
    df = df[(df["time_int"] >= 400) & (df["time_int"] <= 2300)].copy()
    if df.empty:
        print("⚠️ No data inside operating window.")
        return None, None, None

    # Convert timestamps → minutes since midnight
    df.loc[:, "minutes"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    # KDE input data
    X = df["minutes"].values.reshape(-1, 1)

    # KDE model
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)

    # Evaluation grid (1-min resolution)
    minute_grid = np.arange(4 * 60, 23 * 60 + 1).reshape(-1, 1)

    log_density = kde.score_samples(minute_grid)
    density = np.exp(log_density)

    # Convert density → intensity: arrivals per minute
    intensity = density * len(df)

    # Real counts aggregated per minute
    df.loc[:, "minute_bin"] = df["minutes"].astype(int)
    count_per_min = (
        df.groupby("minute_bin").size().reindex(minute_grid.flatten(), fill_value=0)
    )

    return minute_grid.flatten(), intensity, count_per_min


def compute_compensator(intensity):
    """Compute compensator as cumulative integral of intensity."""
    # Each step is 1 minute → integral approximated by cumulative sum
    compensator = np.cumsum(intensity)
    return compensator


def compute_compensator_increments(df, minute_grid, compensator):
    """
    Compute ΔΛ_i = Λ(t_i) − Λ(t_{i−1}) for each event timestamp.
    """

    # Extract actual event minutes
    event_minutes = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute).values

    # Map each event time to index in minute_grid
    grid_to_index = {m: i for i, m in enumerate(minute_grid)}
    comp_at_events = np.array([compensator[grid_to_index[m]] for m in event_minutes])

    # Increments ΔΛ
    deltas = np.diff(comp_at_events)  # length n-1

    return deltas


def nb_pit(counts, mean_vals, k):
    """Dunn–Smyth randomized PIT for Negative Binomial."""
    # NB parameterization: r, p
    var = mean_vals + k * mean_vals**2
    p = mean_vals / var
    r = mean_vals**2 / (var - mean_vals)

    # F(n-1)
    F_lower = nbinom.cdf(counts - 1, r, p)
    # F(n)
    F_upper = nbinom.cdf(counts, r, p)

    # Randomize inside the discrete jump
    u = F_lower + np.random.rand(len(counts)) * (F_upper - F_lower)
    return u


def main(date_str: str, station_code: str):
    # Load data
    folder_path = "data/check_ins/daily"
    file_path = os.path.join(folder_path, f"{date_str}.csv")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    print(f"📂 Loading data for {date_str} — Station {station_code}")
    df = pd.read_csv(
        file_path,
        usecols=["Fecha_Transaccion", "Estacion_Parada"],
        parse_dates=["Fecha_Transaccion"],
    )

    # Filter station
    df_station = df[
        df["Estacion_Parada"].astype(str).str.contains(station_code, case=False)
    ].copy()

    # KDE estimation
    minute_grid, intensity, count_per_min = estimate_intensity_kde(
        df_station, station_code
    )

    if minute_grid is None:
        print("⚠️ No valid data; exiting.")
        return

    # --- Compute Compensator ---
    compensator = compute_compensator(intensity)

    # --- Compute Counting Process ---
    N_t = np.cumsum(count_per_min.values)

    k_mom = kmom_from_counts(count_per_min.values, intensity)
    print(f"Method-of-Moments k = {k_mom:.6f}")
    plot_mean_vs_var(minute_grid, intensity, count_per_min, k_mom)
    k_boot_mean, k_boot_lo, k_boot_hi = bootstrap_k(
        intensity, count_per_min.values, n_boot=500
    )
    print(
        f"Bootstrap k mean={k_boot_mean:.6f}, 95% CI = ({k_boot_lo:.6f}, {k_boot_hi:.6f})"
    )

    # --- Martingale ---
    martingale = N_t - compensator

    df_station.loc[:, "timestamp"] = pd.to_datetime(df_station["Fecha_Transaccion"])
    df_station = df_station.sort_values("timestamp")
    _ = compute_compensator_increments(df_station, minute_grid, compensator)

    # ===== Compute rescaled inter-event times τ_i =====
    event_minutes = (
        df_station["timestamp"].dt.hour * 60 + df_station["timestamp"].dt.minute
    ).values
    grid_to_index = {m: i for i, m in enumerate(minute_grid)}
    lambda_at_events = np.array([compensator[grid_to_index[m]] for m in event_minutes])

    tau = np.diff(lambda_at_events)  # Ogata increments

    ks_exp = kstest(tau, "expon")
    print("\n📌 KS Test for Exp(1) rescaled event times (Ogata):")
    print(ks_exp)

    # ===== Q–Q plot =====
    plt.figure(figsize=(6, 6))
    tau_sorted = np.sort(tau)
    n = len(tau_sorted)
    theoretical = -np.log(1 - np.linspace(0.0001, 0.9999, n))
    plt.plot(theoretical, tau_sorted, "o", markersize=3)
    plt.plot(theoretical, theoretical, "r--")
    plt.xlabel("Theoretical Exp(1) quantiles")
    plt.ylabel("Empirical")
    plt.title("Q–Q Plot of Ogata Rescaled Times")
    plt.show()

    # Fit NB dispersion
    k = fit_nb_dispersion(intensity, count_per_min.values)
    print(f"Estimated NB dispersion k = {k:.3f}")

    # Compute NB PIT
    u = nb_pit(count_per_min.values, intensity, k)

    # ===== Test 1: Autocorrelation of PIT =====
    lags = np.arange(1, 40)
    acf_vals = [np.corrcoef(u[:-lag], u[lag:])[0, 1] for lag in lags]

    markerline, stemlines, baseline = plt.stem(lags, acf_vals)
    plt.setp(markerline, markersize=4)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("ACF of Dunn–Smyth PIT")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.show()

    # ===== Test 2: Ljung–Box test =====
    lb = acorr_ljungbox(u, lags=[10, 20, 30], return_df=True)
    print("\n📌 Ljung–Box Independence Test on PIT")
    print(lb)

    # -- KS test for uniformity
    _, ks_pvalue = kstest(u, "uniform")

    print(f"\n📊 Time-Rescaling PIT KS Test: p-value = {ks_pvalue:.4f}")

    # ==============================
    #           PLOTS
    # ==============================

    # --- Plot KDE + Actual counts ---
    plt.figure(figsize=(14, 6))

    plt.plot(
        minute_grid,
        intensity,
        label="KDE Estimated Intensity (arrivals per minute)",
        linewidth=2,
    )

    plt.bar(
        minute_grid,
        count_per_min.values,
        width=1.0,
        alpha=0.35,
        label="Observed Count per Minute",
    )

    plt.xlabel("Minutes since midnight")
    plt.ylabel("Intensity / Count")
    plt.title(f"KDE Arrival Intensity — Station {station_code} — {date_str}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 12))

    # ---- 1. Counting process N(t) ----
    plt.subplot(3, 1, 1)
    plt.plot(minute_grid, N_t, label="N(t): Counting Process", color="black")
    plt.title(f"Counting Process N(t) — {station_code}")
    plt.xlabel("Minutes since midnight")
    plt.ylabel("N(t)")
    plt.grid(True)

    # ---- 2. Compensator Λ(t) ----
    plt.subplot(3, 1, 2)
    plt.plot(minute_grid, compensator, label="Λ(t): Compensator", color="blue")
    plt.title("Compensator Λ(t)")
    plt.xlabel("Minutes since midnight")
    plt.ylabel("Λ(t)")
    plt.grid(True)

    # ---- 3. Martingale M(t) = N(t) - Λ(t) ----
    plt.subplot(3, 1, 3)
    plt.plot(minute_grid, martingale, label="M(t): Martingale", color="red")
    plt.axhline(0, linestyle="--", color="gray")
    plt.title("Martingale M(t) = N(t) − Λ(t)")
    plt.xlabel("Minutes since midnight")
    plt.ylabel("M(t)")
    plt.grid(True)

    plt.figure(figsize=(12, 10))

    # 1. Histogram of u_i
    plt.subplot(3, 1, 1)
    plt.hist(u, bins=20, density=True, alpha=0.6)
    plt.plot([0, 1], [1, 1], "r--", label="Uniform(0,1) density")
    plt.title("PIT Values Histogram (u_i)")
    plt.xlabel("u_i")
    plt.ylabel("Density")
    plt.legend()

    # 2. QQ Plot
    plt.subplot(3, 1, 2)
    u_sorted = np.sort(u)
    n = len(u)
    uniform_theoretical = np.linspace(0, 1, n)
    plt.plot(uniform_theoretical, u_sorted, "o", markersize=3)
    plt.plot([0, 1], [0, 1], "r--", label="45° line (perfect fit)")
    plt.title("QQ Plot: PIT vs Uniform(0,1)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Empirical Quantiles")
    plt.legend()

    # 3. Empirical CDF
    plt.subplot(3, 1, 3)
    plt.step(u_sorted, np.arange(1, n + 1) / n, where="post", label="Empirical CDF")
    plt.plot([0, 1], [0, 1], "r--", label="Uniform(0,1) CDF")
    plt.title("CDF Comparison")
    plt.xlabel("u")
    plt.ylabel("F(u)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main("20251104", "07107")
