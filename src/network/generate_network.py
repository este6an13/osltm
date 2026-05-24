"""
Generate Transmilenio Station Network from Shapefiles.

Reads station and road (trazado) shapefiles, builds a graph of stations
connected by edges weighted with along-road distances (meters), and exports
the result as JSON and CSV.

Usage:
    uv run python src/network/generate_network.py

Outputs (in data/geometry/output/):
    - network.json        — Full network (nodes + edges)
    - edges.csv           — Edge list with distances
    - nodes.csv           — Node list with attributes
    - network_map.png     — Visual sanity-check plot
    - flagged_stations.csv — Stations that snap poorly to their road
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import linemerge

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STATIONS_SHP = Path("data/geometry/stations/Estación_troncal.shp")
ROADS_SHP = Path("data/geometry/roads/Trazados_Troncales_de_TRANSMILENIO.shp")
CONNECTIONS_SHP = Path("data/geometry/connections/Conexion_operacional.shp")
OUTPUT_DIR = Path("data/geometry/output")

# Snap distance threshold (meters).  Stations farther than this from their
# trazado road will be flagged.
SNAP_WARN_THRESHOLD_M = 100.0

# At each official connection point, we search for the nearest station per
# trazado within this radius to create cross-corridor edges.
CONNECTION_SEARCH_RADIUS_M = 1500.0

# Duplicate merge threshold (meters).  Stations on the same trazado closer
# than this to each other are considered duplicates and merged.
DUPLICATE_THRESHOLD_M = 50.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StationNode:
    id: str
    name: str
    trazado: str
    troncal: str
    tipo: int
    x: float
    y: float
    flagged: bool = False
    flag_reason: str = ""


@dataclass
class NetworkEdge:
    source: str
    target: str
    distance_m: float
    trazado: str | None
    edge_type: str  # "intra_trazado" or "cross_trazado"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_road_geometry(road_geom: Any) -> LineString:
    """Merge MultiLineString into a single LineString.

    Strategy:
    1. Try ``linemerge`` — works when endpoints are exactly shared.
    2. If that still yields a MultiLineString, greedily chain the
       parts by connecting the closest pair of endpoints, effectively
       snapping fragments together into one continuous line.
    """
    if isinstance(road_geom, LineString):
        return road_geom

    merged = linemerge(road_geom)
    if isinstance(merged, LineString):
        return merged

    # Greedy endpoint-chaining for disconnected parts
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return _chain_linestrings(parts)


def _chain_linestrings(parts: list[LineString]) -> LineString:
    """Greedily chain a list of LineStrings into a single LineString.

    Starts with the longest part, then iteratively appends the part
    whose endpoint is closest to the current chain's start or end.
    Parts are reversed as needed so that they connect head-to-tail.
    """
    remaining = list(parts)
    # Start with the longest segment
    remaining.sort(key=lambda g: g.length, reverse=True)
    chain_coords: list[tuple] = list(remaining.pop(0).coords)

    while remaining:
        best_idx = -1
        best_dist = float("inf")
        best_append = "end"  # or "start"
        best_reverse = False

        chain_start = chain_coords[0]
        chain_end = chain_coords[-1]

        for i, part in enumerate(remaining):
            p_start = part.coords[0]
            p_end = part.coords[-1]

            # Try appending part to the END of the chain
            d1 = _pt_dist(chain_end, p_start)   # chain_end -> part_start
            d2 = _pt_dist(chain_end, p_end)      # chain_end -> part_end (reversed)
            # Try prepending part to the START of the chain
            d3 = _pt_dist(chain_start, p_end)    # part_end -> chain_start
            d4 = _pt_dist(chain_start, p_start)  # part_start(rev) -> chain_start

            for d, append, rev in [
                (d1, "end", False),
                (d2, "end", True),
                (d3, "start", False),
                (d4, "start", True),
            ]:
                if d < best_dist:
                    best_dist = d
                    best_idx = i
                    best_append = append
                    best_reverse = rev

        part = remaining.pop(best_idx)
        coords = list(part.coords)
        if best_reverse:
            coords = coords[::-1]

        if best_append == "end":
            chain_coords.extend(coords)
        else:
            chain_coords = coords + chain_coords

    return LineString(chain_coords)


def _pt_dist(a: tuple, b: tuple) -> float:
    """Euclidean distance between two coordinate tuples."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _merge_duplicates(
    station_rows: list[dict],
    threshold_m: float,
) -> list[dict]:
    """Merge stations that share the same ID **or** are within *threshold_m*
    of each other on the same trazado.  Keeps the entry with the shorter
    name (usually the canonical one, e.g. 'Marly' over 'Temporal Marly')."""
    if len(station_rows) <= 1:
        return station_rows

    merged: list[dict] = []
    skip_indices: set[int] = set()

    for i, a in enumerate(station_rows):
        if i in skip_indices:
            continue
        group = [a]
        for j, b in enumerate(station_rows):
            if j <= i or j in skip_indices:
                continue
            same_id = a["id"] == b["id"]
            dist = Point(a["x"], a["y"]).distance(Point(b["x"], b["y"]))
            if same_id or dist < threshold_m:
                group.append(b)
                skip_indices.add(j)

        # Pick the representative: shortest name (canonical)
        representative = min(group, key=lambda s: len(s["name"]))
        merged.append(representative)

    return merged


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_network(
    stations_shp: Path | None = None,
    roads_shp: Path | None = None,
    connections_shp: Path | None = None,
) -> tuple[list[StationNode], list[NetworkEdge]]:
    """Build the station network from shapefiles."""
    stations_shp = stations_shp or STATIONS_SHP
    roads_shp = roads_shp or ROADS_SHP
    connections_shp = connections_shp or CONNECTIONS_SHP

    stations_gdf = gpd.read_file(stations_shp)
    roads_gdf = gpd.read_file(roads_shp)

    log.info("Loaded %d stations and %d roads", len(stations_gdf), len(roads_gdf))

    # Build lookup: trazado_id → road geometry
    road_lookup: dict[str, Any] = {}
    road_meta: dict[str, dict] = {}
    for _, row in roads_gdf.iterrows():
        tz_id = row["id_trazado"]
        road_lookup[tz_id] = row.geometry
        road_meta[tz_id] = {
            "nom_traz": row["nom_traz"],
            "nom_tronc": row["nom_tronc"],
            "ori_traz": row["ori_traz"],
            "fin_traz": row["fin_traz"],
        }

    all_nodes: list[StationNode] = []
    all_edges: list[NetworkEdge] = []
    flagged: list[dict] = []

    trazado_ids = sorted(stations_gdf["id_trazado"].unique())

    for tz_id in trazado_ids:
        tz_stations = stations_gdf[stations_gdf["id_trazado"] == tz_id]

        if tz_id not in road_lookup:
            log.warning("Trazado %s has %d stations but no matching road — skipping",
                        tz_id, len(tz_stations))
            continue

        road_geom = _prepare_road_geometry(road_lookup[tz_id])
        meta = road_meta[tz_id]
        troncal = meta["nom_tronc"]

        log.info("Processing %s (%s) — %d stations",
                 tz_id, meta["nom_traz"], len(tz_stations))

        # Snap each station to the road
        station_rows: list[dict] = []
        for _, st in tz_stations.iterrows():
            pt = st.geometry
            snap_dist = road_geom.distance(pt)
            linear_ref = road_geom.project(pt)

            node_id = str(st["num_est"]).strip()
            name = st["nom_est"]
            is_flagged = snap_dist > SNAP_WARN_THRESHOLD_M

            station_rows.append({
                "id": node_id,
                "name": name,
                "trazado": tz_id,
                "troncal": troncal,
                "tipo": int(st["tipo_esta"]),
                "x": pt.x,
                "y": pt.y,
                "linear_ref": linear_ref,
                "snap_dist": snap_dist,
                "flagged": is_flagged,
            })

            if is_flagged:
                flagged.append({
                    "id": node_id,
                    "name": name,
                    "trazado": tz_id,
                    "snap_distance_m": round(snap_dist, 1),
                    "reason": f"Station is {snap_dist:.0f}m from road geometry",
                })

        # Sort by linear reference
        station_rows.sort(key=lambda s: s["linear_ref"])

        # Merge duplicates (same position within threshold)
        station_rows = _merge_duplicates(station_rows, DUPLICATE_THRESHOLD_M)

        # Create nodes
        for sr in station_rows:
            all_nodes.append(StationNode(
                id=sr["id"],
                name=sr["name"],
                trazado=sr["trazado"],
                troncal=sr["troncal"],
                tipo=sr["tipo"],
                x=sr["x"],
                y=sr["y"],
                flagged=sr["flagged"],
                flag_reason=f"snap_dist={sr['snap_dist']:.0f}m" if sr["flagged"] else "",
            ))

        # Create intra-trazado edges between consecutive stations
        for i in range(len(station_rows) - 1):
            a = station_rows[i]
            b = station_rows[i + 1]
            dist = abs(b["linear_ref"] - a["linear_ref"])

            # Skip zero-distance edges (artifacts of merge)
            if dist < 1.0:
                continue

            all_edges.append(NetworkEdge(
                source=a["id"],
                target=b["id"],
                distance_m=round(dist, 1),
                trazado=tz_id,
                edge_type="intra_trazado",
            ))

    # ------------------------------------------------------------------
    # Cross-trazado connections (driven by Conexion Operacional shapefile)
    # ------------------------------------------------------------------
    log.info("Loading conexiones operacionales from %s...", connections_shp)
    connections_gdf = gpd.read_file(connections_shp)
    log.info("Loaded %d connection points (search radius=%.0fm)",
             len(connections_gdf), CONNECTION_SEARCH_RADIUS_M)

    node_lookup = {n.id: n for n in all_nodes}

    # For each trazado pair, track the best (shortest) edge found across
    # all connection points.  Key = frozenset({tz_a, tz_b}).
    best_cross_edges: dict[frozenset, tuple[StationNode, StationNode, float, str]] = {}

    for _, conn_row in connections_gdf.iterrows():
        conn_pt = conn_row.geometry
        conn_name = conn_row["conexion"]
        conn_id = conn_row["id_conexio"]

        # Find the nearest station per trazado within the search radius
        nearest_per_tz: dict[str, tuple[StationNode, float]] = {}
        for n in all_nodes:
            dist = np.sqrt((n.x - conn_pt.x) ** 2 + (n.y - conn_pt.y) ** 2)
            if dist > CONNECTION_SEARCH_RADIUS_M:
                continue
            if n.trazado not in nearest_per_tz or dist < nearest_per_tz[n.trazado][1]:
                nearest_per_tz[n.trazado] = (n, dist)

        if len(nearest_per_tz) < 2:
            log.warning("  %s (%s): only %d trazado(s) within radius -- skipping",
                        conn_id, conn_name, len(nearest_per_tz))
            continue

        tz_list = sorted(nearest_per_tz.keys())
        log.info("  %s (%s): %d trazados nearby: %s",
                 conn_id, conn_name, len(tz_list), ", ".join(tz_list))

        # For each pair of trazados, check if this connection gives a
        # shorter inter-station distance than what we already have.
        for ti in range(len(tz_list)):
            for tj in range(ti + 1, len(tz_list)):
                node_a, _ = nearest_per_tz[tz_list[ti]]
                node_b, _ = nearest_per_tz[tz_list[tj]]
                dist = np.sqrt(
                    (node_a.x - node_b.x) ** 2 +
                    (node_a.y - node_b.y) ** 2
                )
                key = frozenset({tz_list[ti], tz_list[tj]})
                if key not in best_cross_edges or dist < best_cross_edges[key][2]:
                    best_cross_edges[key] = (node_a, node_b, dist, conn_id)

    # Emit the best edge per trazado pair
    cross_edges_added = 0
    for key, (node_a, node_b, dist, conn_id) in sorted(
        best_cross_edges.items(), key=lambda kv: kv[1][2]
    ):
        all_edges.append(NetworkEdge(
            source=node_a.id,
            target=node_b.id,
            distance_m=round(dist, 1),
            trazado=None,
            edge_type="cross_trazado",
        ))
        cross_edges_added += 1
        log.info("  %s (%s) <-> %s (%s) = %.0fm [via %s]",
                 node_a.name, node_a.trazado,
                 node_b.name, node_b.trazado, dist, conn_id)

    log.info("Added %d cross-trazado edges from %d connection points",
             cross_edges_added, len(connections_gdf))

    # Save flagged stations
    if flagged:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flagged).to_csv(OUTPUT_DIR / "flagged_stations.csv", index=False)
        log.info("Flagged %d stations with poor snap → flagged_stations.csv", len(flagged))

    return all_nodes, all_edges


def export_json(nodes: list[StationNode], edges: list[NetworkEdge], path: Path) -> None:
    """Export network as JSON."""
    data = {
        "metadata": {
            "description": "Transmilenio station network generated from official shapefiles",
            "crs": "EPSG:3116",
            "distance_unit": "meters",
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "intra_trazado_edges": sum(1 for e in edges if e.edge_type == "intra_trazado"),
            "cross_trazado_edges": sum(1 for e in edges if e.edge_type == "cross_trazado"),
        },
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Exported JSON → %s", path)


def export_csv(nodes: list[StationNode], edges: list[NetworkEdge], out_dir: Path) -> None:
    """Export nodes and edges as CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_df = pd.DataFrame([asdict(n) for n in nodes])
    nodes_df.to_csv(out_dir / "nodes.csv", index=False, encoding="utf-8")

    edges_df = pd.DataFrame([asdict(e) for e in edges])
    edges_df.to_csv(out_dir / "edges.csv", index=False, encoding="utf-8")

    log.info("Exported CSV → %s/nodes.csv, edges.csv", out_dir)


def plot_network(
    nodes: list[StationNode],
    edges: list[NetworkEdge],
    out_path: Path,
) -> None:
    """Generate a visual map of the network."""
    G = nx.Graph()
    for n in nodes:
        G.add_node(n.id, pos=(n.x, n.y), name=n.name, trazado=n.trazado)
    for e in edges:
        G.add_edge(e.source, e.target, distance=e.distance_m,
                   edge_type=e.edge_type, trazado=e.trazado)

    pos = nx.get_node_attributes(G, "pos")

    # Color nodes by trazado
    trazados = sorted(set(n.trazado for n in nodes))
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(trazados))
    tz_color = {tz: cmap(i) for i, tz in enumerate(trazados)}
    node_colors = [tz_color[G.nodes[n]["trazado"]] for n in G.nodes]

    # Separate intra and cross edges
    intra_edges = [(e.source, e.target) for e in edges if e.edge_type == "intra_trazado"]
    cross_edges = [(e.source, e.target) for e in edges if e.edge_type == "cross_trazado"]

    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_aspect("equal")

    # Draw intra edges
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges,
                           edge_color="#555555", width=1.5, alpha=0.7, ax=ax)
    # Draw cross edges
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges,
                           edge_color="#e74c3c", width=1.0, alpha=0.5,
                           style="dashed", ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=30, alpha=0.9, ax=ax)

    # Label portals (tipo=1)
    portal_nodes = {n.id: n.name for n in nodes if n.tipo == 1}
    portal_labels = {nid: name for nid, name in portal_nodes.items() if nid in pos}
    nx.draw_networkx_labels(G, pos, labels=portal_labels,
                            font_size=6, font_weight="bold", ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#555555", lw=1.5, label="Intra-trazado"),
        Line2D([0], [0], color="#e74c3c", lw=1.0, linestyle="--", label="Cross-trazado"),
    ]
    for tz, color in tz_color.items():
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=6, label=tz)
        )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              framealpha=0.9, ncol=2)

    ax.set_title("Transmilenio Station Network", fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m, EPSG:3116)")
    ax.set_ylabel("Northing (m, EPSG:3116)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved network map → %s", out_path)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_network(nodes: list[StationNode], edges: list[NetworkEdge]) -> None:
    """Run basic sanity checks on the generated network."""
    log.info("=" * 50)
    log.info("VERIFICATION")
    log.info("=" * 50)

    node_ids = {n.id for n in nodes}
    log.info("Total nodes: %d", len(nodes))
    log.info("Total edges: %d", len(edges))
    log.info("  Intra-trazado: %d", sum(1 for e in edges if e.edge_type == "intra_trazado"))
    log.info("  Cross-trazado: %d", sum(1 for e in edges if e.edge_type == "cross_trazado"))

    # Check all edge endpoints exist
    missing = set()
    for e in edges:
        if e.source not in node_ids:
            missing.add(e.source)
        if e.target not in node_ids:
            missing.add(e.target)
    if missing:
        log.error("❌ Edges reference %d missing nodes: %s", len(missing), missing)
    else:
        log.info("✓ All edge endpoints exist in node set")

    # Check for negative distances
    neg = [e for e in edges if e.distance_m < 0]
    if neg:
        log.error("❌ %d edges have negative distances", len(neg))
    else:
        log.info("✓ No negative distances")

    # Check connectivity
    G = nx.Graph()
    G.add_nodes_from(n.id for n in nodes)
    G.add_edges_from((e.source, e.target) for e in edges)
    components = list(nx.connected_components(G))
    log.info("Connected components: %d", len(components))
    if len(components) > 1:
        for i, comp in enumerate(components):
            names = [n.name for n in nodes if n.id in comp]
            log.info("  Component %d: %d nodes (e.g. %s)", i, len(comp),
                     ", ".join(names[:3]))

    # Total network length
    total_km = sum(e.distance_m for e in edges if e.edge_type == "intra_trazado") / 1000
    log.info("Total intra-trazado network length: %.1f km", total_km)

    # Nodes without edges
    nodes_in_edges = set()
    for e in edges:
        nodes_in_edges.add(e.source)
        nodes_in_edges.add(e.target)
    isolated = node_ids - nodes_in_edges
    if isolated:
        iso_names = [n.name for n in nodes if n.id in isolated]
        log.warning("⚠ %d isolated nodes (no edges): %s", len(isolated), iso_names)
    else:
        log.info("✓ All nodes have at least one edge")

    # Flagged stations
    flagged_nodes = [n for n in nodes if n.flagged]
    if flagged_nodes:
        log.warning("⚠ %d stations flagged for poor road snap:", len(flagged_nodes))
        for fn in flagged_nodes:
            log.warning("  %s (%s) — %s", fn.name, fn.trazado, fn.flag_reason)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(generate_plot: bool = True) -> None:
    nodes, edges = build_network()
    verify_network(nodes, edges)

    export_json(nodes, edges, OUTPUT_DIR / "network.json")
    export_csv(nodes, edges, OUTPUT_DIR)
    if generate_plot:
        plot_network(nodes, edges, OUTPUT_DIR / "network_map.png")

    log.info("Done! Output in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
