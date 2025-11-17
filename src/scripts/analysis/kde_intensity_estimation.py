import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import kstest
from sklearn.neighbors import KernelDensity


def cameron_trivedi_overdispersion_test(y):
    """
    Robust Cameron–Trivedi (1990) test for Poisson overdispersion.
    Works even when Poisson fitted mean is constant.
    """
    y = np.asarray(y)
    X = np.ones((len(y), 1))

    # Fit Poisson
    pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    mu = pois.fittedvalues

    # If mu has essentially no variation → test not identifiable
    if np.allclose(mu, mu[0]):
        print("⚠️ Poisson fitted mean is constant — overdispersion test unavailable.")
        return np.nan, np.nan, 1.0, pois

    # Pearson residuals
    r = (y - mu) / np.sqrt(mu)

    # Auxiliary regression: r^2 = α * μ + c
    aux_y = r**2
    aux_X = sm.add_constant(mu)  # now 2 columns

    aux = sm.OLS(aux_y, aux_X).fit()

    # If regression crashed or degenerate
    if len(aux.params) < 2:
        print("⚠️ Degenerate auxiliary regression — cannot estimate overdispersion.")
        return np.nan, np.nan, 1.0, pois

    alpha = aux.params[1]
    alpha_se = aux.bse[1]

    # z-test ( α > 0 one-sided )
    z = alpha / alpha_se
    p = 1 - stats.norm.cdf(z)

    return alpha, z, p, pois


def test_negative_binomial(count_per_min):
    y = count_per_min.values

    print("\n===============================")
    print("📊 NEGATIVE BINOMIAL OVERDISPERSION TESTS")
    print("===============================\n")

    alpha, z, p, pois = cameron_trivedi_overdispersion_test(y)

    if np.isnan(alpha):
        print("⚠️ Overdispersion test could not be computed (degenerate data).")
    else:
        print("📌 Cameron–Trivedi Overdispersion Test")
        print(f"   α estimate: {alpha:.4f}")
        print(f"   z-statistic: {z:.4f}")
        print(f"   p-value: {p:.6f}")

        if p < 0.05:
            print("👉 Reject Poisson — overdispersion detected.")
        else:
            print("✅ No significant overdispersion.")

    # Fit NB model even if test fails — statsmodels can still estimate it
    nb = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.NegativeBinomial()).fit()

    print("\n📉 Model Comparison (AIC):")
    print(f"   Poisson AIC: {pois.aic:.2f}")
    print(f"   NB AIC:      {nb.aic:.2f}")

    # Residual comparison
    plt.figure(figsize=(14, 6))
    plt.plot(pois.resid_pearson, label="Poisson Pearson Residuals", alpha=0.7)
    plt.plot(nb.resid_pearson, label="NB Pearson Residuals", alpha=0.7)
    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Residual Comparison: Poisson vs Negative Binomial")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pois, nb


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


def pit_uniform(deltas):
    """Convert exponential(1) increments to Uniform(0,1)."""

    return 1 - np.exp(-deltas)


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

    # --- Martingale ---
    martingale = N_t - compensator

    pois_model, nb_model = test_negative_binomial(count_per_min)

    df_station.loc[:, "timestamp"] = pd.to_datetime(df_station["Fecha_Transaccion"])
    df_station = df_station.sort_values("timestamp")
    deltas = compute_compensator_increments(df_station, minute_grid, compensator)

    u = pit_uniform(deltas)

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

    plt.tight_layout()
    plt.show()

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
