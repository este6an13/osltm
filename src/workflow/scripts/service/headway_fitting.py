"""
Bus Arrival Headway Fitting Script (Service Rate Characterization)

Extracts planned route frequencies, simulates a Transit Delay Mixture Model
representing driver deviations and traffic congestion delays, and fits three 
candidate continuous probability distributions (Gamma, Erlang, Log-Normal) via MLE.
Outputs AIC/BIC comparison metrics to justify service times statistically.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Filter warnings for cleaner console output
warnings.filterwarnings("ignore")


def parse_frequency_string(freq_str: str) -> float:
    """
    Parse strings like '10 min', '4 min', '---' into float minutes.
    Falls back to a default of 5.0 minutes if unparsable or empty.
    """
    if not isinstance(freq_str, str) or "---" in freq_str or not freq_str.strip():
        return 5.0
    
    clean = freq_str.lower().replace("min", "").replace(" ", "").strip()
    try:
        return float(clean)
    except ValueError:
        return 5.0


def simulate_transit_delays(mu_H: float, cv: float, n_samples: int = 1000) -> np.ndarray:
    """
    Generates a realistic Transit Delay Mixture Model:
      H = max(0.1, scheduled_headway + driver_wobble + traffic_congestion_delay)
    
    where:
      - driver_wobble ~ Normal(0, sigma_w^2) represents schedule deviations
      - traffic_congestion_delay ~ Exponential(theta_d) represents traffic delays
    """
    # Wobble is a small fraction of scheduled mean (e.g. 10%)
    sigma_w = 0.10 * mu_H
    
    # Exponential traffic delay represents the rest of the target variation
    # Total variance Var(H) = Var(w) + Var(d) = sigma_w^2 + theta_d^2
    # We want target standard deviation to be cv * mu_H
    target_var = (cv * mu_H) ** 2
    wobble_var = sigma_w ** 2
    
    if target_var > wobble_var:
        theta_d = np.sqrt(target_var - wobble_var)
    else:
        theta_d = 0.05 * mu_H  # tiny fallback delay if cv is extremely small

    # Draw samples
    np.random.seed(42)  # For reproducibility
    w = np.random.normal(0.0, sigma_w, n_samples)
    d = np.random.exponential(theta_d, n_samples)
    
    H = mu_H + w + d
    
    # Clip at 0.1 minutes (6 seconds) to prevent non-positive arrivals
    H = np.clip(H, 0.1, None)
    
    return H


def fit_distributions(H: np.ndarray) -> dict:
    """
    Fit Gamma, Erlang, and Log-Normal distributions to the data using MLE.
    Returns optimal parameters, Log-Likelihood, AIC, and BIC.
    """
    n = len(H)
    results = {}
    
    # Fix loc=0 for all fits to keep support strictly positive (physically correct)
    
    # ── 1. Gamma Fit ───────────────────────────────────────────────────────
    # shape (k), loc, scale (theta)
    k_gam, _, theta_gam = stats.gamma.fit(H, floc=0)
    log_lik_gam = np.sum(stats.gamma.logpdf(H, k_gam, scale=theta_gam))
    aic_gam = 2 * 2 - 2 * log_lik_gam
    bic_gam = np.log(n) * 2 - 2 * log_lik_gam
    
    results["gamma"] = {
        "params": {"shape": float(k_gam), "scale": float(theta_gam)},
        "log_likelihood": float(log_lik_gam),
        "aic": float(aic_gam),
        "bic": float(bic_gam),
    }

    # ── 2. Erlang Fit ──────────────────────────────────────────────────────
    # Erlang is Gamma with integer shape k. We round the Gamma shape and re-fit scale.
    k_erl = max(1, int(round(k_gam)))
    # For a fixed shape k, the MLE scale is simply: mean / k
    theta_erl = np.mean(H) / k_erl
    log_lik_erl = np.sum(stats.erlang.logpdf(H, k_erl, scale=theta_erl))
    aic_erl = 2 * 2 - 2 * log_lik_erl
    # Note: k_erl is technically discrete, but counts as a parameter
    bic_erl = np.log(n) * 2 - 2 * log_lik_erl
    
    results["erlang"] = {
        "params": {"shape_k": int(k_erl), "scale": float(theta_erl)},
        "log_likelihood": float(log_lik_erl),
        "aic": float(aic_erl),
        "bic": float(bic_erl),
    }

    # ── 3. Log-Normal Fit ──────────────────────────────────────────────────
    # s (sigma), loc, scale (exp(mu))
    s_logn, _, scale_logn = stats.lognorm.fit(H, floc=0)
    mu_logn = np.log(scale_logn)
    log_lik_logn = np.sum(stats.lognorm.logpdf(H, s_logn, scale=scale_logn))
    aic_logn = 2 * 2 - 2 * log_lik_logn
    bic_logn = np.log(n) * 2 - 2 * log_lik_logn
    
    results["lognormal"] = {
        "params": {"sigma": float(s_logn), "mu": float(mu_logn), "scale": float(scale_logn)},
        "log_likelihood": float(log_lik_logn),
        "aic": float(aic_logn),
        "bic": float(bic_logn),
    }
    
    return results


def generate_comparison_plot(
    H: np.ndarray,
    fits: dict,
    route_name: str,
    period: str,
    output_path: Path
) -> None:
    """Plot headway histogram with overlaid probability density function curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Histogram of raw simulated headways
    count, bins, _ = ax.hist(H, bins=35, density=True, color="#d1d5db", edgecolor="#9ca3af", alpha=0.6, label="Simulated Headways")
    
    # Generate smooth x-axis grid for continuous plotting
    x = np.linspace(0.1, np.max(H) * 1.1, 300)
    
    # 2. Plot Gamma
    gam = fits["gamma"]["params"]
    ax.plot(x, stats.gamma.pdf(x, gam["shape"], scale=gam["scale"]), color="#1e3a8a", linewidth=2.5, label="Fitted Gamma")
    
    # 3. Plot Erlang
    erl = fits["erlang"]["params"]
    ax.plot(x, stats.erlang.pdf(x, erl["shape_k"], scale=erl["scale"]), color="#15803d", linewidth=2.5, linestyle="--", label=f"Fitted Erlang (k={erl['shape_k']})")
    
    # 4. Plot Log-Normal
    logn = fits["lognormal"]["params"]
    ax.plot(x, stats.lognorm.pdf(x, logn["sigma"], scale=logn["scale"]), color="#b91c1c", linewidth=2.5, linestyle=":", label="Fitted Log-Normal")
    
    ax.set_xlabel("Bus Headway Interval (Minutes)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(f"Headway Distribution Fitting: Route {route_name} ({period.capitalize()} Period)\nScheduled Interval: {np.mean(H):.2f} min", fontsize=13, fontweight="bold")
    
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOTTED] Static comparison plot saved to: {output_path.name}")


def run_headway_fitting(
    route_name: list[str] | str,
    period: str = "peak",
    cv: float = 0.25,
    n_samples: int = 1000,
    output_dir: Optional[Path] = None,
    frequencies_csv: Optional[Path] = None,
) -> dict:
    """Simulates bus arrivals, performs MLE fits, and generates reports."""
    # ── 1. Load Scheduled Frequency ───────────────────────────────────────
    if frequencies_csv is None:
        frequencies_csv = Path("data/routes/transmilenio_frequencies.csv")
    
    if not frequencies_csv.exists():
        # Fallback lookups
        possible_paths = [
            Path("osltm/data/routes/transmilenio_frequencies.csv"),
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "routes" / "transmilenio_frequencies.csv",
            Path("d:/dequi/repositories/osltm/data/routes/transmilenio_frequencies.csv"),
        ]
        for p in possible_paths:
            if p.exists():
                frequencies_csv = p
                break

    if not frequencies_csv.exists():
        raise FileNotFoundError(f"Scheduled frequencies CSV not found at: {frequencies_csv.resolve()}")

    df = pd.read_csv(frequencies_csv)
    
    if isinstance(route_name, str):
        routes = [route_name]
    else:
        routes = list(route_name)

    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    if output_dir is None:
        output_dir = PROJECT_ROOT / "src" / "workflow" / "results" / "headway_fitting"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
            
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    
    for r in routes:
        print(f"\n>>> Processing Route {r}...")
        # Search for matching name (case-insensitive)
        match_df = df[df["name"].str.strip().str.upper() == r.strip().upper()]
        
        if match_df.empty:
            print(f"[WARNING] Route '{r}' not found in frequencies database. Falling back to default scheduled interval of 6.0 minutes.")
            mu_H = 6.0
        else:
            # Load peak or off-peak frequency string
            col = "frequency_peak" if period.lower() == "peak" else "frequency_offpeak"
            freq_str = match_df.iloc[0][col]
            mu_H = parse_frequency_string(freq_str)
            print(f"[LOADED] Loaded scheduled frequency for Route {r} ({period}): {freq_str} -> {mu_H:.1f} minutes")

        # ── 2. Simulate Transit Delays (Proxy Telemetry) ──────────────────────
        print(f"[SIMULATING] Simulating {n_samples} headways from Transit Delay Mixture Model (Cv={cv:.2f})...")
        H = simulate_transit_delays(mu_H, cv, n_samples)
        
        # Save simulated samples to CSV so the results viewer/API can read it
        samples_file = output_dir / f"simulated_headways_{r}.csv"
        pd.DataFrame({"headway_minutes": H}).to_csv(samples_file, index=False)
        print(f"[SAVED] Raw simulated samples saved to: {samples_file}")

        # Save single fallback file for backwards compatibility
        if r == routes[0] or len(routes) == 1:
            sh_file = output_dir / "simulated_headways.csv"
            pd.DataFrame({"headway_minutes": H}).to_csv(sh_file, index=False)

        # ── 3. Perform MLE fits ────────────────────────────────────────────────
        print("[FITTING] Fitting Gamma, Erlang, and Log-Normal models via scipy MLE...")
        fits = fit_distributions(H)
        
        # Save fits to JSON
        meta_file = output_dir / f"fitted_headway_report_{r}.json"
        report_data = {
            "route_name": r,
            "period": period,
            "cv": cv,
            "scheduled_mean": mu_H,
            "simulated_mean": float(np.mean(H)),
            "simulated_std": float(np.std(H)),
            "fits": fits
        }
        with open(meta_file, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"[SAVED] Goodness-of-fit parameters report exported to: {meta_file}")

        if r == routes[0] or len(routes) == 1:
            default_meta_file = output_dir / "fitted_headway_report.json"
            with open(default_meta_file, "w") as f:
                json.dump(report_data, f, indent=2)

        # ── 4. Generate Plot ──────────────────────────────────────────────────
        plot_file = output_dir / f"fitted_headways_comparison_{r}.png"
        generate_comparison_plot(H, fits, r, period, plot_file)
        
        if r == routes[0] or len(routes) == 1:
            default_plot_file = output_dir / "fitted_headways_comparison.png"
            generate_comparison_plot(H, fits, r, period, default_plot_file)
        
        # Print summary table
        print(f"\n[RESULTS] {r} GOODNESS-OF-FIT CRITERIA:")
        print(f"{'Distribution':<15} | {'Log-Likelihood':<15} | {'AIC':<12} | {'BIC':<12}")
        print("-" * 62)
        for dist, dat in fits.items():
            print(f"{dist.capitalize():<15} | {dat['log_likelihood']:<15.3f} | {dat['aic']:<12.3f} | {dat['bic']:<12.3f}")
        
        # Determine winner
        winner = min(fits.keys(), key=lambda k: fits[k]["aic"])
        print(f"\n[WINNER] Best-fitting distribution (by AIC/BIC): {winner.upper()}")

        results[r] = {
            "samples": H,
            "fits": fits,
            "winner": winner,
            "scheduled_mean": mu_H
        }

    # ── 5. Run Traversal Simulation ──────────────────────────────────────
    print("\n>>> Simulating 2D bus traversals (View 1 & View 2)...")
    simulate_traversal_and_serialize(routes, period, cv, output_dir, results)

    return results


def simulate_traversal_and_serialize(
    routes: list[str],
    period: str,
    cv: float,
    output_dir: Path,
    report_data_dict: dict
) -> None:
    """
    Simulates bus traversals for both View 1 (Fitted) and View 2 (Physical)
    and serializes the timeline to traversal_simulation.json.
    """
    import math
    # 1. Load nodes.csv
    nodes_csv = Path("data/geometry/output/nodes.csv")
    if not nodes_csv.exists():
        possible_paths = [
            Path("osltm/data/geometry/output/nodes.csv"),
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "geometry" / "output" / "nodes.csv",
            Path("d:/dequi/repositories/osltm/data/geometry/output/nodes.csv")
        ]
        for p in possible_paths:
            if p.exists():
                nodes_csv = p
                break
                
    def clean_id(val):
        try:
            if pd.isnull(val):
                return ""
            return str(int(float(val))).strip().zfill(5)
        except Exception:
            return str(val).strip().zfill(5)

    stations_db = {}
    if nodes_csv.exists():
        try:
            df_nodes = pd.read_csv(nodes_csv)
            for _, row in df_nodes.iterrows():
                sid = clean_id(row["id"])
                stations_db[sid] = {
                    "station_id": sid,
                    "name": str(row["name"]).strip(),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "troncal": str(row["troncal"]).strip() if pd.notnull(row["troncal"]) else "Other"
                }
        except Exception as e:
            print(f"[WARNING] Failed to parse nodes.csv: {e}")
    else:
        print(f"[WARNING] nodes.csv not found at: {nodes_csv.resolve()}")

    # 1.2 Load edges.csv
    edges_list = []
    edges_csv = Path("data/geometry/output/edges.csv")
    if not edges_csv.exists():
        possible_paths = [
            Path("osltm/data/geometry/output/edges.csv"),
            Path(__file__).resolve().parents[4] / "data" / "geometry" / "output" / "edges.csv",
            Path("d:/dequi/repositories/osltm/data/geometry/output/edges.csv")
        ]
        for p in possible_paths:
            if p.exists():
                edges_csv = p
                break
    if edges_csv.exists():
        try:
            df_edges = pd.read_csv(edges_csv)
            for _, row in df_edges.iterrows():
                src = clean_id(row["source"])
                tgt = clean_id(row["target"])
                if src in stations_db and tgt in stations_db:
                    edges_list.append({
                        "source": src,
                        "target": tgt,
                        "edge_type": str(row["edge_type"]) if pd.notnull(row["edge_type"]) else "intra_trazado"
                    })
        except Exception as e:
            print(f"[WARNING] Failed to parse edges.csv: {e}")

    # 2. Load transmilenio_routes_stations.csv
    routes_csv = Path("data/routes/transmilenio_routes_stations.csv")
    if not routes_csv.exists():
        possible_paths = [
            Path("osltm/data/routes/transmilenio_routes_stations.csv"),
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "routes" / "transmilenio_routes_stations.csv",
            Path("d:/dequi/repositories/osltm/data/routes/transmilenio_routes_stations.csv")
        ]
        for p in possible_paths:
            if p.exists():
                routes_csv = p
                break

    routes_stations_df = None
    if routes_csv.exists():
        try:
            routes_stations_df = pd.read_csv(routes_csv)
        except Exception as e:
            print(f"[WARNING] Failed to parse routes_stations CSV: {e}")

    routes_payload = {}
    
    for r in routes:
        if routes_stations_df is None:
            continue
            
        # Find matching rows
        match_df = routes_stations_df[routes_stations_df["route_code"].astype(str).str.strip().str.upper() == r.strip().upper()]
        if match_df.empty:
            print(f"[WARNING] Route '{r}' sequence not found in routes CSV. Skipping traversal simulation.")
            continue
            
        # Identify the route_id with the maximum number of stations (primary sequence)
        route_ids = match_df["route_id"].unique()
        best_route_id = None
        best_count = 0
        for rid in route_ids:
            cnt = len(match_df[match_df["route_id"] == rid])
            if cnt > best_count:
                best_count = cnt
                best_route_id = rid
                
        best_df = match_df[match_df["route_id"] == best_route_id].sort_values("station_sequence")
        
        # Build station sequence trace
        route_stations_seq = []
        route_color = "#3b82f6" # default blue
        
        for _, row in best_df.iterrows():
            sid = clean_id(row["station_id"])
            if "route_color" in row and pd.notnull(row["route_color"]):
                route_color = str(row["route_color"]).strip()
            if sid in stations_db:
                route_stations_seq.append(sid)
                
        if len(route_stations_seq) < 2:
            print(f"[WARNING] Route '{r}' has less than 2 matched stations in geometry. Skipping.")
            continue
            
        # Fetch fits parameters
        route_report = report_data_dict.get(r)
        if not route_report:
            print(f"[WARNING] No fitting fits found for route '{r}'. Skipping.")
            continue
            
        fits = route_report["fits"]
        winner = route_report["winner"]
        
        # ─── VIEW 1: Fitted MLE Model (Nominal Corridor) ───
        np.random.seed(42)
        n_buses = 30
        
        if winner == "gamma":
            shape = fits["gamma"]["params"]["shape"]
            scale = fits["gamma"]["params"]["scale"]
            headways = stats.gamma.rvs(shape, scale=scale, size=n_buses, random_state=42)
        elif winner == "erlang":
            shape_k = fits["erlang"]["params"]["shape_k"]
            scale = fits["erlang"]["params"]["scale"]
            headways = stats.erlang.rvs(shape_k, scale=scale, size=n_buses, random_state=42)
        else: # lognormal
            sigma = fits["lognormal"]["params"]["sigma"]
            scale = fits["lognormal"]["params"]["scale"]
            headways = stats.lognorm.rvs(sigma, scale=scale, size=n_buses, random_state=42)
            
        # Keep headways positive and realistic
        headways = np.clip(headways, 0.2, None)
        dispatch_times_fit = np.cumsum(headways) * 60.0 # in seconds
        
        buses_fitted = []
        base_speed = 30.0 / 3.6 # 30 km/h in m/s = 8.33
        base_dwell = 30.0 # 30 seconds constant
        
        for b in range(n_buses):
            bus_id = f"{r}_FIT_{b+1}"
            t_curr = dispatch_times_fit[b]
            timeline = []
            
            for i, sid in enumerate(route_stations_seq):
                t_arr = t_curr
                t_dep = t_arr + base_dwell
                timeline.append({
                    "station_id": sid,
                    "arrival": round(t_arr, 1),
                    "departure": round(t_dep, 1)
                })
                
                t_curr = t_dep
                if i < len(route_stations_seq) - 1:
                    s_curr = stations_db[sid]
                    s_next = stations_db[route_stations_seq[i+1]]
                    dist = math.sqrt((s_next["x"] - s_curr["x"])**2 + (s_next["y"] - s_curr["y"])**2)
                    travel_time = dist / base_speed
                    t_curr += travel_time
                    
            buses_fitted.append({
                "bus_id": bus_id,
                "timeline": timeline
            })
            
        # ─── VIEW 2: Physical Model (Stochastic Degradation) ───
        scheduled_mean = route_report["scheduled_mean"]
        dispatch_times_phys = np.arange(1, n_buses + 1) * scheduled_mean * 60.0 # regular schedule
        
        buses_physical = []
        
        for b in range(n_buses):
            bus_id = f"{r}_PHYS_{b+1}"
            v_b = np.random.normal(30.0 / 3.6, 1.8 / 3.6)
            v_b = np.clip(v_b, 6.0, 11.0)
            
            t_curr = dispatch_times_phys[b]
            timeline = []
            
            for i, sid in enumerate(route_stations_seq):
                t_arr = t_curr
                dwell = np.random.normal(30.0, 5.0)
                dwell = max(15.0, dwell)
                t_dep = t_arr + dwell
                
                timeline.append({
                    "station_id": sid,
                    "arrival": round(t_arr, 1),
                    "departure": round(t_dep, 1)
                })
                
                t_curr = t_dep
                if i < len(route_stations_seq) - 1:
                    s_curr = stations_db[sid]
                    s_next = stations_db[route_stations_seq[i+1]]
                    dist = math.sqrt((s_next["x"] - s_curr["x"])**2 + (s_next["y"] - s_curr["y"])**2)
                    travel_time_base = dist / v_b
                    
                    m = travel_time_base
                    s = cv * m * 0.5
                    
                    if s > 0:
                        var_normal = math.log(1 + (s / m)**2)
                        mu_normal = math.log(m) - var_normal / 2
                        travel_time = math.exp(np.random.normal(mu_normal, math.sqrt(var_normal)))
                    else:
                        travel_time = m
                        
                    t_curr += travel_time
                    
            buses_physical.append({
                "bus_id": bus_id,
                "timeline": timeline
            })
            
        routes_payload[r] = {
            "route_code": r,
            "color": route_color,
            "stations": route_stations_seq,
            "views": {
                "fitted": {
                    "description": f"Fitted Model: Stochastic dispatches using best-fitting {winner.upper()} model, deterministic stable corridor travel.",
                    "buses": buses_fitted
                },
                "physical": {
                    "description": "Physical Model: Perfectly uniform schedule dispatches, experiencing en-route driver perturbations, stochastic dwells, and log-normal traffic delay noise.",
                    "buses": buses_physical
                }
            }
        }
        
    payload = {
        "metadata": {
            "period": period,
            "cv": cv,
            "simulated_routes": list(routes_payload.keys())
        },
        "stations": stations_db,
        "edges": edges_list,
        "routes": routes_payload
    }
    
    output_file = output_dir / "traversal_simulation.json"
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[TRAVERSAL SIMULATOR] Traversal simulation written to: {output_file}")
    
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    fallback_file = PROJECT_ROOT / "src" / "workflow" / "results" / "headway_fitting" / "traversal_simulation.json"
    fallback_file.parent.mkdir(parents=True, exist_ok=True)
    with open(fallback_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[TRAVERSAL SIMULATOR] Traversal simulation fallback written to: {fallback_file}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stochastically perturbs planned bus frequencies and fits Gamma, Erlang, and Log-Normal models."
    )
    parser.add_argument(
        "--route_name",
        type=str,
        nargs="+",
        default=["B12"],
        help="Transit service name(s) to model (default: B12)",
    )
    parser.add_argument(
        "--period",
        choices=["peak", "offpeak"],
        default="peak",
        help="Day period to model (default: peak)",
    )
    parser.add_argument(
        "--cv",
        type=float,
        default=0.25,
        help="Stochastic traffic delay Coefficient of Variation (default: 0.25)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of headway samples to draw (default: 1000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/headway_fitting",
        help="Output folder (default: src/workflow/results/headway_fitting)",
    )
    parser.add_argument(
        "--frequencies",
        type=str,
        default="data/routes/transmilenio_frequencies.csv",
        help="Path to route frequencies database CSV",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="src/workflow/params.json",
        help="Path to params.json (ignored but accepted for runner compatibility)",
    )

    args = parser.parse_args()
    
    run_headway_fitting(
        route_name=args.route_name,
        period=args.period,
        cv=args.cv,
        n_samples=args.n_samples,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        frequencies_csv=Path(args.frequencies) if args.frequencies else None,
    )
    
    print("\n[FINISHED] Week 2 Bus Headway Stochastic Fitting successfully finished!")
