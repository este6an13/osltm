"""
LGCP Backtesting — Cutoff-aware Simulation & Diagnostics

For each (station, day_type), performs prior-based and/or posterior-based
simulation from the fitted LGCP, then compares against post-cutoff (test)
observed data via envelope plots.

Prior simulation (from Phase 2 kernel params):
    z ~ N(mu, C_smooth)           GP prior (day-level log-intensity draw)
    N_k ~ Poisson(exp(z_k))       bin counts

Posterior simulation (from Phase 2 kernel + Phase 3 posterior):
    z ~ N(z_map, H^{-1})         Laplace posterior
    N_k ~ Poisson(exp(z_k))       bin counts

Both paths use only pre-cutoff data for fitting (mu and kernel/posterior
estimates), and evaluate on post-cutoff data.

Outputs:
  - lgcp_simulations_{ct}_{sim_type}.csv   — binned counts per simulated day
  - lgcp_envelope_{s}_{d}_{ct}_{sim_type}.png  — diagnostics envelope plots
"""

import json
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.workflow.data_loader import load_data, load_persisted_data

DATE_TYPE_LABELS = {"WD": "Weekday", "SA": "Saturday", "SU": "Sunday", "HO": "Holiday"}


def kernel_se(t: np.ndarray, sigma2: float, ell: float, eta2: float) -> np.ndarray:
    diffs = t[:, None] - t[None, :]
    K = sigma2 * np.exp(-0.5 * (diffs / ell) ** 2)
    K += eta2 * np.eye(len(t))
    return K


def kernel_matern32(t: np.ndarray, sigma2: float, ell: float, eta2: float) -> np.ndarray:
    diffs = np.abs(t[:, None] - t[None, :])
    r = np.sqrt(3) * diffs / ell
    K = sigma2 * (1 + r) * np.exp(-r)
    K += eta2 * np.eye(len(t))
    return K


def time_int_to_hours(t_int: int) -> float:
    return t_int // 100 + (t_int % 100) / 60.0


def sort_time_columns(cols: list[str]) -> list[str]:
    return sorted(cols, key=lambda c: int(c.replace("t_", "")))


def simulate_one_day_prior(
    mu: np.ndarray,
    L: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one day from the GP prior: z ~ N(mu, C_smooth), N_k ~ Poisson(exp(z_k))."""
    eps = rng.standard_normal(len(mu))
    z = mu + L @ eps
    lam = np.exp(z)
    return rng.poisson(lam).astype(int)


def simulate_one_day_posterior(
    z_map: np.ndarray,
    L_H: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one day from the Laplace posterior: z ~ N(z_map, H^{-1}), N_k ~ Poisson(exp(z_k)).

    H = L_H @ L_H.T  (numpy Cholesky convention: L lower-triangular).
    To draw v ~ N(0, H^{-1}) we need v = L_H^{-T} eps = solve(L_H.T, eps),
    because Cov(v) = L_H^{-T} L_H^{-1} = (L_H L_H^T)^{-1} = H^{-1}.
    """
    eps = rng.standard_normal(len(z_map))
    z = z_map + np.linalg.solve(L_H.T, eps)   # correct upper-triangular backsolve
    lam = np.exp(z)
    return rng.poisson(lam).astype(int)


def plot_envelope(
    test_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    station_code: str,
    day_type: str,
    time_cols: list[str],
    time_hours: np.ndarray,
    output_dir: Path,
    count_type: str,
    sim_type: str,
):
    """Plot simulation envelopes vs actual test data."""
    if sim_df.empty or test_df.empty:
        return

    s_test = test_df[(test_df["station_code"] == station_code) & (test_df["date_type"] == day_type)]
    s_sim = sim_df[(sim_df["station_code"] == station_code) & (sim_df["day_type"] == day_type)]

    if s_test.empty or s_sim.empty:
        return

    sim_counts = s_sim[time_cols].values
    sim_mean = np.mean(sim_counts, axis=0)
    sim_p05 = np.percentile(sim_counts, 5, axis=0)
    sim_p95 = np.percentile(sim_counts, 95, axis=0)
    sim_p25 = np.percentile(sim_counts, 25, axis=0)
    sim_p75 = np.percentile(sim_counts, 75, axis=0)

    plt.figure(figsize=(12, 6))
    plt.fill_between(time_hours, sim_p05, sim_p95, color="blue", alpha=0.1, label="90% Envelope (Sim)")
    plt.fill_between(time_hours, sim_p25, sim_p75, color="blue", alpha=0.2, label="50% Envelope (Sim)")
    plt.plot(time_hours, sim_mean, "b-", linewidth=2, label="Simulated Mean")

    actual_counts = s_test[time_cols].values
    for i, day in enumerate(actual_counts):
        label = "Actual Test Days" if i == 0 else None
        plt.plot(time_hours, day, "k-", alpha=0.3, linewidth=0.8, label=label)

    actual_mean = np.mean(actual_counts, axis=0)
    plt.plot(time_hours, actual_mean, "r--", linewidth=1.5, label="Actual Test Mean")

    plt.xlabel("Time of Day (hours)")
    plt.ylabel("Passenger Counts")
    dt_label = DATE_TYPE_LABELS.get(day_type, day_type)

    if sim_type == "prior":
        title_line = "LGCP Prior Simulation"
    elif sim_type == "posterior":
        title_line = "LGCP Posterior Simulation"
    else:
        title_line = f"LGCP Simulation ({sim_type})"

    plt.title(f"{title_line} — {station_code} ({dt_label})\nCounts: {count_type.capitalize()}", fontweight="bold")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)

    fname = output_dir / f"lgcp_envelope_{station_code}_{day_type}_{count_type}_{sim_type}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def run_simulation_prior(
    df_train: pd.DataFrame,
    kernel_sel: pd.DataFrame,
    time_cols: list[str],
    time_hours: np.ndarray,
    n_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run prior-based simulation for all station/day_type pairs."""
    all_sim = []

    for _, krow in kernel_sel.iterrows():
        station_code = krow["station_code"]
        day_type = krow["day_type"]
        kernel_name = krow["kernel"]
        sigma2 = krow["sigma2"]
        ell = krow["ell_hours"]

        sdf = df_train[(df_train["station_code"] == station_code) & (df_train["date_type"] == day_type)]
        if len(sdf) < 5:
            continue

        counts_all = sdf[time_cols].fillna(0).values.astype(float)
        mean_counts = counts_all.mean(axis=0)
        mu = np.log(mean_counts + 0.5)

        kernel_fn = kernel_se if kernel_name == "SE" else kernel_matern32
        C_smooth = kernel_fn(time_hours, sigma2, ell, 0.0)
        C_smooth += 1e-6 * np.eye(len(time_hours))

        try:
            L = np.linalg.cholesky(C_smooth)
        except np.linalg.LinAlgError:
            C_smooth += 1e-4 * np.eye(len(time_hours))
            L = np.linalg.cholesky(C_smooth)

        for d in range(n_days):
            counts = simulate_one_day_prior(mu, L, rng)
            res = {"station_code": station_code, "day_type": day_type, "sim_day": d}
            for i, col in enumerate(time_cols):
                res[col] = int(counts[i])
            all_sim.append(res)

    if all_sim:
        return pd.DataFrame(all_sim)
    return pd.DataFrame(columns=["station_code", "day_type", "sim_day"] + list(time_cols))


def run_simulation_posterior(
    kernel_sel: pd.DataFrame,
    posterior_df: pd.DataFrame,
    time_cols: list[str],
    time_hours: np.ndarray,
    n_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run posterior-based simulation for all station/day_type pairs."""
    all_sim = []
    K = len(time_hours)
    delta_t = 1.0

    for _, krow in kernel_sel.iterrows():
        station_code = krow["station_code"]
        day_type = krow["day_type"]
        kernel_name = krow["kernel"]
        sigma2 = krow["sigma2"]
        ell = krow["ell_hours"]
        eta2 = krow["eta2"]

        sp = posterior_df[
            (posterior_df["station_code"] == station_code)
            & (posterior_df["day_type"] == day_type)
        ]
        if sp.empty:
            continue

        # Sort by time_bin so z_map[k] aligns with time_hours[k]
        sp = sp.copy()
        sp["_t_int"] = sp["time_bin"].str.replace("t_", "", regex=False).astype(int)
        sp = sp.sort_values("_t_int").reset_index(drop=True)
        z_map = sp["z_posterior_mean"].values.astype(float)

        kernel_fn = kernel_se if kernel_name == "SE" else kernel_matern32
        C = kernel_fn(time_hours, sigma2, ell, eta2)
        C += 1e-6 * np.eye(K)

        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            C_inv = np.linalg.pinv(C)

        W = np.diag(np.exp(z_map) * delta_t)
        H = W + C_inv
        H += 1e-6 * np.eye(K)

        try:
            L_H = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            H += 1e-4 * np.eye(K)
            L_H = np.linalg.cholesky(H)

        for d in range(n_days):
            counts = simulate_one_day_posterior(z_map, L_H, rng)
            res = {"station_code": station_code, "day_type": day_type, "sim_day": d}
            for i, col in enumerate(time_cols):
                res[col] = int(counts[i])
            all_sim.append(res)

    if all_sim:
        return pd.DataFrame(all_sim)
    return pd.DataFrame(columns=["station_code", "day_type", "sim_day"] + list(time_cols))


def run_lgcp_simulate_backtest(
    count_type: Literal["checkins", "checkouts"] = "checkins",
    sim_type: Literal["prior", "posterior", "both"] = "both",
    n_days: int = 30,
    params_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    phase2_dir: Optional[Path] = None,
    bayesian_dir: Optional[Path] = None,
    cutoff_date: Optional[str] = None,
    station_codes: Optional[list[str]] = None,
    day_types: Optional[list[str]] = None,
    seed: int = 42,
) -> dict:
    """
    Run cutoff-aware LGCP simulation and diagnostics.

    Args:
        count_type: "checkins" or "checkouts"
        sim_type: "prior", "posterior", or "both"
        n_days: Number of simulated days per (station, day_type)
        cutoff_date: YYYY-MM-DD cutoff (default from params.json)
        phase2_dir: Path to Phase 2 (two-stage) results
        bayesian_dir: Path to Phase 3 (bayesian) results (needed for posterior)
        station_codes: Optional station filter
        day_types: Optional day-type filter (WD, SA, SU, HO)
        seed: RNG seed
    """
    if params_path is None:
        params_path = Path("src/workflow/params.json")

    with open(params_path) as f:
        params = json.load(f)

    step4 = params.get("step4", {})
    time_min = step4.get("time_min", 400)
    time_max = step4.get("time_max", 2300)
    time_step = step4.get("time_step", 15)

    lgcp_params = params.get("lgcp", {})
    if cutoff_date is None:
        cutoff_date = lgcp_params.get("cutoff_date", "2025-11-30")
    if output_dir is None:
        output_dir = Path(lgcp_params.get("output_dir", "src/workflow/results/lgcp_backtest"))
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    persistence_dir = params_path.parent
    _, sampled_stations = load_persisted_data(persistence_dir)
    if sampled_stations:
        available = [s["code"] for s in sampled_stations]
        station_codes = (
            [sc for sc in station_codes if sc in available]
            if station_codes
            else available
        )
    elif station_codes is None:
        raise ValueError("No station codes available.")

    if phase2_dir is None:
        phase2_dir = Path("src/workflow/results/lgcp_twostage")
    phase2_dir = Path(phase2_dir)

    kernel_csv = phase2_dir / f"lgcp_kernel_params_{count_type}.csv"
    if not kernel_csv.exists():
        raise FileNotFoundError(f"Phase 2 kernel params not found: {kernel_csv}")
    kernel_df = pd.read_csv(kernel_csv)
    kernel_df["station_code"] = kernel_df["station_code"].astype(str).str.zfill(5)
    kernel_sel = kernel_df[kernel_df["is_selected"] == True].copy()

    if station_codes and len(station_codes) > 0:
        kernel_sel = kernel_sel[kernel_sel["station_code"].isin(station_codes)]
    if day_types and len(day_types) > 0:
        kernel_sel = kernel_sel[kernel_sel["day_type"].isin(day_types)]

    print(f"Loaded {len(kernel_sel)} selected kernels from Phase 2")

    if sim_type in ("posterior", "both"):
        if bayesian_dir is None:
            # Identify pipeline_id and exp_id from phase2_dir (if fully-scoped)
            pipeline_id = None
            exp_id_part = None
            for part in phase2_dir.parts:
                if part.startswith("pipeline_"):
                    pipeline_id = part
                elif part.startswith("exp_"):
                    exp_id_part = part

            # Strategy 1: sibling directory — ONLY when phase2_dir is a full exp-scoped path
            # (lgcp_twostage/pipeline_X/exp_Y -> lgcp_bayesian/pipeline_X/exp_Y)
            # Avoids accidentally picking up the bare-root lgcp_bayesian/ file.
            posterior_csv = None
            if pipeline_id and exp_id_part:
                sibling = Path(str(phase2_dir).replace("lgcp_twostage", "lgcp_bayesian"))
                sibling_csv = sibling / f"lgcp_posterior_params_{count_type}.csv"
                if sibling_csv.exists():
                    posterior_csv = sibling_csv
                    print(f"   Auto-detected posterior CSV (sibling exp): {posterior_csv}")

            # Strategy 2: glob most recent exp_* under the same pipeline
            if posterior_csv is None and pipeline_id:
                bayesian_candidates = sorted(
                    Path("src/workflow/results/lgcp_bayesian").glob(
                        f"{pipeline_id}/exp_*/lgcp_posterior_params_{count_type}.csv"
                    ),
                    reverse=True,
                )
                if bayesian_candidates:
                    posterior_csv = bayesian_candidates[0]
                    print(f"   Auto-detected posterior CSV (pipeline glob): {posterior_csv}")

            if posterior_csv is None:
                hint = (
                    f"lgcp_bayesian/{pipeline_id}/exp_*/lgcp_posterior_params_{count_type}.csv"
                    if pipeline_id else "lgcp_bayesian/<pipeline_id>/exp_*/"
                )
                raise FileNotFoundError(
                    f"Phase 3 posterior params not found. Searched: {hint}. "
                    f"Run step2_bayesian for pipeline '{pipeline_id or 'unknown'}' first, "
                    f"or pass --bayesian_dir explicitly."
                )
        else:
            bayesian_dir = Path(bayesian_dir)
            posterior_csv = bayesian_dir / f"lgcp_posterior_params_{count_type}.csv"
            if not posterior_csv.exists():
                raise FileNotFoundError(f"Phase 3 posterior params not found: {posterior_csv}")
        posterior_df = pd.read_csv(posterior_csv)
        posterior_df["station_code"] = posterior_df["station_code"].astype(str).str.zfill(5)
        print(f"   Loaded posterior CSV: {posterior_csv} ({len(posterior_df)} rows)")
    else:
        posterior_df = pd.DataFrame()

    print(f"Loading {count_type} data...")
    data = load_data(
        station_codes=station_codes,
        include_checkins=(count_type == "checkins"),
        include_checkouts=(count_type == "checkouts"),
        time_min=time_min, time_max=time_max, time_step=time_step,
    )
    df = data[count_type]
    df["station_code"] = df["station_code"].astype(str).str.zfill(5)

    time_cols = sort_time_columns([c for c in df.columns if c.startswith("t_")])

    df["date_str"] = df.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
    train_df = df[df["date_str"] <= cutoff_date].copy()
    test_df = df[df["date_str"] > cutoff_date].copy()

    print(f"   Training records: {len(train_df)}, Test records: {len(test_df)} (cutoff={cutoff_date})")

    time_hours = np.array([time_int_to_hours(int(c.replace("t_", ""))) for c in time_cols])

    rng = np.random.default_rng(seed)
    results = {}

    if sim_type in ("prior", "both"):
        print(f"\nRunning prior-based simulation ({n_days} days per station/day_type)...")
        sim_prior = run_simulation_prior(
            train_df, kernel_sel, time_cols, time_hours, n_days, rng
        )
        csv_prior = output_dir / f"lgcp_simulations_{count_type}_prior.csv"
        sim_prior.to_csv(csv_prior, index=False)
        print(f"Saved {len(sim_prior)} prior-simulated days to {csv_prior}")
        results["prior"] = sim_prior

        if not test_df.empty:
            print("Generating prior envelope plots...")
            stations_to_plot = test_df["station_code"].unique()
            dt_to_plot = day_types if day_types and len(day_types) > 0 else test_df["date_type"].unique()
            for sc in stations_to_plot:
                for dt in dt_to_plot:
                    plot_envelope(test_df, sim_prior, sc, dt, time_cols, time_hours, output_dir, count_type, "prior")

    if sim_type in ("posterior", "both"):
        print(f"\nRunning posterior-based simulation ({n_days} days per station/day_type)...")
        sim_post = run_simulation_posterior(
            kernel_sel, posterior_df, time_cols, time_hours, n_days, rng
        )
        csv_post = output_dir / f"lgcp_simulations_{count_type}_posterior.csv"
        sim_post.to_csv(csv_post, index=False)
        print(f"Saved {len(sim_post)} posterior-simulated days to {csv_post}")
        results["posterior"] = sim_post

        if not test_df.empty:
            print("Generating posterior envelope plots...")
            stations_to_plot = test_df["station_code"].unique()
            dt_to_plot = day_types if day_types and len(day_types) > 0 else test_df["date_type"].unique()
            for sc in stations_to_plot:
                for dt in dt_to_plot:
                    plot_envelope(test_df, sim_post, sc, dt, time_cols, time_hours, output_dir, count_type, "posterior")

    if test_df.empty:
        print("\nNo data found after cutoff date for backtesting. Skipping diagnostics.")
    else:
        print(f"\nDiagnostics completed. Plots saved to {output_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cutoff-aware LGCP simulation and backtesting diagnostics"
    )
    parser.add_argument("--count_type", choices=["checkins", "checkouts"], default="checkins")
    parser.add_argument("--sim_type", choices=["prior", "posterior", "both"], default="both")
    parser.add_argument("--n_days", type=int, default=30, help="Simulated days per station/day_type")
    parser.add_argument("--params", type=str, default="src/workflow/params.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--phase2_dir", type=str, default="src/workflow/results/lgcp_twostage")
    parser.add_argument("--bayesian_dir", type=str, default=None, help="Path to Phase 3 (Bayesian) results dir. Auto-detected from --phase2_dir if omitted.")
    parser.add_argument("--cutoff_date", type=str, default=None, help="YYYY-MM-DD cutoff")
    parser.add_argument("--stations", nargs="+", default=None)
    parser.add_argument("--day_types", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_lgcp_simulate_backtest(
        count_type=args.count_type,
        sim_type=args.sim_type,
        n_days=args.n_days,
        params_path=Path(args.params) if args.params else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        phase2_dir=Path(args.phase2_dir) if args.phase2_dir else None,
        bayesian_dir=Path(args.bayesian_dir) if args.bayesian_dir else None,
        cutoff_date=args.cutoff_date,
        station_codes=args.stations,
        day_types=args.day_types,
        seed=args.seed,
    )
