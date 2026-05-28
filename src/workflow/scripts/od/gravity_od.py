"""
Spatial Interaction Model (Gravity/Entropy-Maximization Model) & IPF OD Estimation

Fits a doubly-constrained spatial interaction model using the Iterative Proportional
Fitting (IPF) algorithm to estimate time-varying passenger routing probabilities
P(j | i, t) across the TransMilenio network using aggregated 15-minute counts
and spatial shortest-path along-road distances.

Outputs:
  - estimated_od_probabilities.csv   — origin, destination, time_bin, flow, probability
  - gravity_od_flow_heatmap_am.png   — flow heatmap at 07:30 AM (morning peak)
  - gravity_od_flow_heatmap_pm.png   — flow heatmap at 06:00 PM (evening peak)
  - portal_commute_reversal.png      — entries vs exits over time for major portals
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.workflow.data_loader import load_data, load_persisted_data, generate_time_columns

# Standard date-type labels matching the rest of the project
DATE_TYPE_LABELS = {
    "WD": "Weekday",
    "SA": "Saturday",
    "SU": "Sunday",
    "HO": "Holiday",
}


def load_spatial_network(nodes_csv: Path, edges_csv: Path) -> tuple[dict[str, tuple[float, float]], np.ndarray, list[str]]:
    """
    Load network nodes and edges, compute pairwise along-road shortest-path distances.
    If some stations are physically disconnected, falls back to spatial Euclidean distance.

    Returns:
        coords: dict of station_code -> (x, y)
        dist_matrix: K x K numpy array of pairwise distances (meters)
        station_codes: sorted list of K station codes
    """
    # Load nodes (stations) and edges (routes)
    nodes_df = pd.read_csv(nodes_csv, dtype={"id": str, "trazado": str})
    edges_df = pd.read_csv(edges_csv, dtype={"source": str, "target": str, "trazado": str})

    # Standardize codes to 5-char zero-padded strings
    nodes_df["code"] = nodes_df["id"].str.zfill(5)
    edges_df["source"] = edges_df["source"].str.zfill(5)
    edges_df["target"] = edges_df["target"].str.zfill(5)

    # Coordinates lookup
    coords = {
        row["code"]: (row["x"], row["y"])
        for _, row in nodes_df.iterrows()
    }

    station_codes = sorted(nodes_df["code"].unique())
    K = len(station_codes)
    code_to_idx = {code: i for i, code in enumerate(station_codes)}

    # Build NetworkX Graph
    G = nx.Graph()
    G.add_nodes_from(station_codes)
    for _, edge in edges_df.iterrows():
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["distance_m"]
        )

    print(f"🕸️ Built spatial graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    components = list(nx.connected_components(G))
    print(f"   Connected components: {len(components)}")

    # Compute shortest path distances using Dijkstra
    dist_matrix = np.zeros((K, K))
    for i, src in enumerate(station_codes):
        # Calculate single-source Dijkstra path lengths
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        
        for j, dest in enumerate(station_codes):
            if dest in lengths:
                dist_matrix[i, j] = lengths[dest]
            else:
                # Fallback to Euclidean distance if in separate components
                xi, yi = coords[src]
                xj, yj = coords[dest]
                euc = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                dist_matrix[i, j] = euc

    # Set self-distance to 0
    np.fill_diagonal(dist_matrix, 0.0)

    return coords, dist_matrix, station_codes


def run_doubly_constrained_ipf(
    U: np.ndarray,
    V: np.ndarray,
    D: np.ndarray,
    gamma: float = 0.0001,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Runs the Iterative Proportional Fitting (IPF) algorithm to solve the doubly-constrained
    spatial interaction model:
        T_ij = A_i * B_j * U_i * V_j * exp(-gamma * d_ij)

    Args:
        U: vector of origin counts (K,)
        V: vector of destination counts (K,)
        D: cost/distance matrix (K, K)
        gamma: spatial decay parameter
        max_iter: maximum number of balancing iterations
        tol: convergence tolerance

    Returns:
        K x K flow matrix T
    """
    K = len(U)
    
    # 1. Handle edge cases (empty counts)
    sum_U = U.sum()
    sum_V = V.sum()
    if sum_U == 0 or sum_V == 0:
        return np.zeros((K, K))

    # Scale destinations (V) to match origins (U) exactly for flow conservation: sum_i U_i = sum_j V_j
    V_scaled = V * (sum_U / sum_V)

    # 2. Compute seed matrix W (thermodynamic distance decay friction)
    W = np.exp(-gamma * D)
    
    # Do not allow travel to oneself (diagonal is 0)
    np.fill_diagonal(W, 0.0)

    # 3. Balancing Iterations
    T = W.copy()
    
    for iteration in range(max_iter):
        # Step 3a: Row Balancing (Origins constraint)
        row_sums = T.sum(axis=1)
        row_factor = np.zeros_like(row_sums)
        valid_rows = row_sums > 0
        row_factor[valid_rows] = U[valid_rows] / row_sums[valid_rows]
        T = T * row_factor[:, None]

        # Step 3b: Column Balancing (Destinations constraint)
        col_sums = T.sum(axis=0)
        col_factor = np.zeros_like(col_sums)
        valid_cols = col_sums > 0
        col_factor[valid_cols] = V_scaled[valid_cols] / col_sums[valid_cols]
        T = T * col_factor[None, :]

        # Check convergence
        current_rows = T.sum(axis=1)
        current_cols = T.sum(axis=0)
        
        row_err = np.max(np.abs(current_rows - U))
        col_err = np.max(np.abs(current_cols - V_scaled))
        
        if max(row_err, col_err) < tol:
            break

    return T


def plot_flow_heatmap(
    T: np.ndarray,
    station_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot a heatmap of the estimated station-to-station flows."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot log-flows for visual clarity (due to massive scale differences)
    log_T = np.log10(T + 1.0)
    
    im = ax.imshow(log_T, cmap="YlOrRd", aspect="equal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Estimated passenger flow (log10 scale)", fontsize=11)

    # Set ticks for major stations (portals/major intermediates)
    major_indices = [i for i, name in enumerate(station_names) if "PORTAL" in name.upper() or i % 10 == 0]
    major_labels = [station_names[i] for i in major_indices]

    ax.set_xticks(major_indices)
    ax.set_xticklabels(major_labels, rotation=90, fontsize=8)
    ax.set_yticks(major_indices)
    ax.set_yticklabels(major_labels, fontsize=8)

    ax.set_xlabel("Alighting Station (Destination)", fontsize=12)
    ax.set_ylabel("Boarding Station (Origin)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  💾 Heatmap plotted to {output_path.name}")


def plot_commute_reversal(
    checkins_df: pd.DataFrame,
    checkouts_df: pd.DataFrame,
    time_cols: list[str],
    time_hours: np.ndarray,
    station_codes: list[str],
    station_names_map: dict[str, str],
    output_path: Path,
) -> None:
    """Plot entries vs exits over time for major portals to show commute reversal."""
    # Find the top 3 portals with the most check-ins
    portals = [sc for sc in station_codes if "PORTAL" in station_names_map.get(sc, "").upper()]
    if not portals:
        # Fallback to top stations
        portals = station_codes[:3]
    else:
        # Keep top 3 portals by total checkins
        portals = sorted(portals, key=lambda sc: checkins_df[checkins_df["station_code"] == sc][time_cols].fillna(0).values.sum(), reverse=True)[:3]

    fig, axes = plt.subplots(len(portals), 1, figsize=(10, 3 * len(portals)), sharex=True)
    if len(portals) == 1:
        axes = [axes]

    for idx, sc in enumerate(portals):
        ax = axes[idx]
        name = station_names_map.get(sc, sc)

        # Average weekday entries and exits
        in_series = checkins_df[checkins_df["station_code"] == sc][time_cols].fillna(0).mean(axis=0).values
        out_series = checkouts_df[checkouts_df["station_code"] == sc][time_cols].fillna(0).mean(axis=0).values

        ax.plot(time_hours, in_series, "b-", linewidth=2, label="Check-ins (Entries)")
        ax.plot(time_hours, out_series, "r--", linewidth=2, label="Check-outs (Exits)")
        
        ax.set_ylabel("Avg Passengers / 15-min", fontsize=10)
        ax.set_title(f"{sc} — {name}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(time_hours[0], time_hours[-1])
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time of Day (Hours)", fontsize=11)
    fig.suptitle("Commute Pattern Reversal at Major Portals (Weekday Averages)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  💾 Commute reversal plotted to {output_path.name}")


def run_gravity_od(
    day_type: str = "WD",
    cutoff_date: Optional[str] = None,
    gamma: float = 0.0001,
    min_days: int = 5,
    output_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    station_codes_filter: Optional[list[str]] = None,
) -> dict:
    """Runs the full Spatial Interaction / IPF OD estimation pipeline."""
    # ── 1. Load Parameters ────────────────────────────────────────────────
    if params_path is None:
        params_path = Path("src/workflow/params.json")
    with open(params_path) as f:
        params = json.load(f)

    step4_params = params.get("step4", {})
    time_min = step4_params.get("time_min", 400)
    time_max = step4_params.get("time_max", 2300)
    time_step = step4_params.get("time_step", 15)

    persistence_dir = params_path.parent / "data"
    sampled_dates, sampled_stations = load_persisted_data(persistence_dir)
    if not sampled_dates:
        raise ValueError("No sampled dates found. Run workflow steps 1-4 first.")

    # Stations list
    if sampled_stations:
        available_stations = [s["code"] for s in sampled_stations]
        station_codes = (
            [sc for sc in station_codes_filter if sc in available_stations]
            if station_codes_filter
            else available_stations
        )
    else:
        raise ValueError("No sampled stations found in data directory.")

    if output_dir is None:
        output_dir = Path("src/workflow/results/gravity_od")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Load Spatial Graph & Pairwise Distances ─────────────────────────
    nodes_csv = Path("data/geometry/output/nodes.csv")
    edges_csv = Path("data/geometry/output/edges.csv")
    
    if not nodes_csv.exists() or not edges_csv.exists():
        raise FileNotFoundError(
            f"Spatial outputs not found at {nodes_csv.parent}. "
            "Please run network generator step 5 first: "
            "`uv run python -m src.network.generate_network`"
        )

    coords, dist_matrix, network_station_codes = load_spatial_network(nodes_csv, edges_csv)

    # Keep only stations present in BOTH the network and count samples
    common_stations = sorted(list(set(station_codes) & set(network_station_codes)))
    if not common_stations:
        raise ValueError("No overlap between sampled stations and spatial network station codes!")
    
    print(f"✅ Filtered to {len(common_stations)} stations present in both network and samples.")

    # Slice distance matrix to match common_stations
    net_indices = [network_station_codes.index(sc) for sc in common_stations]
    D = dist_matrix[np.ix_(net_indices, net_indices)]

    # ── 3. Load Count Data ────────────────────────────────────────────────
    print(f"📊 Loading count data...")
    data = load_data(
        station_codes=common_stations,
        include_checkins=True,
        include_checkouts=True,
        time_min=time_min, time_max=time_max, time_step=time_step,
    )
    
    checkins_df = data["checkins"]
    checkouts_df = data["checkouts"]

    if cutoff_date is not None:
        # Apply cutoff date filter
        for df_key in ["checkins", "checkouts"]:
            df = data[df_key]
            df["date_str"] = df.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1)
            data[df_key] = df[df["date_str"] <= cutoff_date].copy()
        checkins_df = data["checkins"]
        checkouts_df = data["checkouts"]
        print(f"   Applied cutoff filter <= {cutoff_date}")

    time_cols = sorted([c for c in checkins_df.columns if c.startswith("t_")], key=lambda c: int(c.replace("t_", "")))
    time_hours = np.array([int(tc.replace("t_", "")) // 100 + (int(tc.replace("t_", "")) % 100) / 60.0 for tc in time_cols])

    # Name map
    station_names_map = {}
    for _, row in checkins_df.drop_duplicates("station_code").iterrows():
        station_names_map[row["station_code"]] = row["station_name"]

    # Filter out stations with insufficient days
    valid_stations = []
    for sc in common_stations:
        n_days_in = len(checkins_df[(checkins_df["station_code"] == sc) & (checkins_df["date_type"] == day_type)])
        n_days_out = len(checkouts_df[(checkouts_df["station_code"] == sc) & (checkouts_df["date_type"] == day_type)])
        if n_days_in >= min_days and n_days_out >= min_days:
            valid_stations.append(sc)

    if len(valid_stations) < 2:
        raise ValueError(
            f"Not enough stations with >= {min_days} days for day_type {day_type}. "
            "Reduce --min_days or add more data."
        )

    print(f"📊 Valid stations with >= {min_days} days: {len(valid_stations)} (out of {len(common_stations)})")
    
    # Re-slice distance matrix to match valid_stations
    val_indices = [common_stations.index(sc) for sc in valid_stations]
    D_val = D[np.ix_(val_indices, val_indices)]
    
    # Keep names in order of valid_stations
    valid_names = [station_names_map.get(sc, sc) for sc in valid_stations]

    # ── 4. Compute Time-Varying Doubly-Constrained IPF ─────────────────────
    print(f"\n🔬 Running IPF Gravity Model for each 15-min time bin (Day-Type: {DATE_TYPE_LABELS.get(day_type, day_type)})...")

    # Group count dfs by valid stations and selected day type
    in_grouped = checkins_df[(checkins_df["station_code"].isin(valid_stations)) & (checkins_df["date_type"] == day_type)]
    out_grouped = checkouts_df[(checkouts_df["station_code"].isin(valid_stations)) & (checkouts_df["date_type"] == day_type)]

    # Compute average entries and exits per station per time bin
    # We pivot or group to ensure we align with valid_stations order
    in_means = in_grouped.groupby("station_code")[time_cols].mean().reindex(valid_stations).fillna(0.0).values
    out_means = out_grouped.groupby("station_code")[time_cols].mean().reindex(valid_stations).fillna(0.0).values

    # Output rows list
    output_rows = []
    
    # Store dynamic flows for plotting
    flow_matrices = {}

    for t_idx, time_col in enumerate(time_cols):
        # Marginal vectors for this time bin
        U = in_means[:, t_idx]   # Check-ins (Origins)
        V = out_means[:, t_idx]  # Check-outs (Destinations)

        # Solve IPF
        T = run_doubly_constrained_ipf(U, V, D_val, gamma=gamma, max_iter=150)
        flow_matrices[time_col] = T

        # Calculate transition probabilities: P(j | i, t) = T_ij / U_i
        # Avoid division by zero
        row_sums = T.sum(axis=1)
        prob_matrix = np.zeros_like(T)
        for i in range(len(valid_stations)):
            if row_sums[i] > 0:
                prob_matrix[i, :] = T[i, :] / row_sums[i]

        # Verify flow conservation
        sum_U = U.sum()
        if sum_U > 0:
            row_err = np.max(np.abs(T.sum(axis=1) - U))
            # Column error compares to scaled destinations
            V_scaled = V * (sum_U / V.sum()) if V.sum() > 0 else V
            col_err = np.max(np.abs(T.sum(axis=0) - V_scaled))
            
            # Print verification for peak hours
            if time_col in ["t_730", "t_1800"]:
                print(f"   Bin {time_col} Balanced: row_err={row_err:.4f}, col_err={col_err:.4f} (Total flow: {sum_U:.1f})")

        # Save to records
        for i, src in enumerate(valid_stations):
            for j, dest in enumerate(valid_stations):
                output_rows.append({
                    "origin_code": src,
                    "origin_name": valid_names[i],
                    "destination_code": dest,
                    "destination_name": valid_names[j],
                    "time_bin": time_col,
                    "distance_m": D_val[i, j],
                    "estimated_flow": T[i, j],
                    "routing_probability": prob_matrix[i, j],
                })

    # Save to CSV
    output_df = pd.DataFrame(output_rows)
    csv_file = output_dir / "estimated_od_probabilities.csv"
    output_df.to_csv(csv_file, index=False)
    print(f"\n💾 Saved dynamic OD probabilities to {csv_file}")

    # ── 5. Visualizations ─────────────────────────────────────────────────
    print("📈 Generating analytical plots...")
    
    # 5a. Heatmap AM Peak (07:30 AM)
    if "t_730" in flow_matrices:
        plot_flow_heatmap(
            flow_matrices["t_730"],
            valid_names,
            f"Estimated Passenger Flow Heatmap — Morning Peak (07:30 AM)\nDay-Type: {DATE_TYPE_LABELS.get(day_type, day_type)}",
            output_dir / "gravity_od_flow_heatmap_am.png"
        )

    # 5b. Heatmap PM Peak (06:00 PM)
    if "t_1800" in flow_matrices:
        plot_flow_heatmap(
            flow_matrices["t_1800"],
            valid_names,
            f"Estimated Passenger Flow Heatmap — Evening Peak (06:00 PM)\nDay-Type: {DATE_TYPE_LABELS.get(day_type, day_type)}",
            output_dir / "gravity_od_flow_heatmap_pm.png"
        )

    # 5c. Commute pattern reversal line chart
    plot_commute_reversal(
        checkins_df,
        checkouts_df,
        time_cols,
        time_hours,
        valid_stations,
        station_names_map,
        output_dir / "portal_commute_reversal.png"
    )

    print(f"\n✅ Completed Spatial Interaction & IPF OD Estimation for {day_type}!")
    
    return {
        "od_df": output_df,
        "valid_stations": valid_stations,
        "flow_matrices": flow_matrices
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimates time-varying passenger routing probabilities P(j | i, t) using a Doubly-Constrained Gravity Model & IPF."
    )
    parser.add_argument(
        "--day_type",
        choices=["WD", "SA", "SU", "HO"],
        default="WD",
        help="Day-type to model (default: WD)",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="YYYY-MM-DD cutoff to filter historical validation data (dates <= cutoff)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0001,
        help="Friction decay parameter per meter (default: 0.0001)",
    )
    parser.add_argument(
        "--min_days",
        type=int,
        default=5,
        help="Minimum replicate days required per station/day-type (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/gravity_od",
        help="Output folder (default: src/workflow/results/gravity_od)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="src/workflow/params.json",
        help="Path to params.json (default: src/workflow/params.json)",
    )
    parser.add_argument(
        "--stations",
        type=str,
        nargs="+",
        help="Optional list of station codes to run on",
    )

    args = parser.parse_args()

    results = run_gravity_od(
        day_type=args.day_type,
        cutoff_date=args.cutoff_date,
        gamma=args.gamma,
        min_days=args.min_days,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        params_path=Path(args.params),
        station_codes_filter=args.stations,
    )
    
    print("\n🎉 Spatial Interaction & IPF OD Estimation Pipeline Finished successfully!")
