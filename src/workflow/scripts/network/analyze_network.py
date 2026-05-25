"""
Analyze and visualize the Transmilenio station network from the database.

Retrieves nodes and edges, builds a NetworkX graph, calculates centrality
metrics (degree, betweenness, closeness), and generates static PNG plots.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.db.session_v2 import SessionLocal
from src.repo.v2.network.repository import NetworkRepository

log = logging.getLogger(__name__)

BASE_OUTPUT_DIR = Path("src/workflow/results/network")


def load_graph_from_db() -> nx.Graph:
    """Read network from SQLite database and build a NetworkX graph."""
    db = SessionLocal()
    try:
        repo = NetworkRepository(db)
        if not repo.has_data():
            raise ValueError("No network data found in the database. Run the pipeline Step 5 first.")

        nodes = repo.get_all_nodes()
        edges = repo.get_all_edges()

        log.info("Loaded %d nodes and %d edges from DB", len(nodes), len(edges))

        G = nx.Graph()
        
        # Add nodes with attributes
        for n in nodes:
            G.add_node(
                n.station_id,
                name=n.name,
                trazado=n.trazado,
                troncal=n.troncal,
                tipo=n.tipo,
                x=n.x,
                y=n.y,
            )

        # Add edges with attributes
        for e in edges:
            G.add_edge(
                e.source,
                e.target,
                distance_m=e.distance_m,
                trazado=e.trazado,
                edge_type=e.edge_type,
            )
        # Remove disconnected nodes (isolates)
        isolates = list(nx.isolates(G))
        if isolates:
            log.info("Removing %d disconnected stations: %s", len(isolates), isolates)
            G.remove_nodes_from(isolates)

        return G
    finally:
        db.close()


def compute_metrics(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute centrality metrics for the graph."""
    log.info("Computing network metrics...")
    
    degree = nx.degree_centrality(G)
    
    # betweenness usually considers weight as 'cost', so distance is appropriate
    betweenness = nx.betweenness_centrality(G, weight="distance_m")
    
    # closeness centrality uses distance
    closeness = nx.closeness_centrality(G, distance="distance_m")

    # Attach metrics to nodes for easy extraction later
    nx.set_node_attributes(G, degree, "degree_centrality")
    nx.set_node_attributes(G, betweenness, "betweenness_centrality")
    nx.set_node_attributes(G, closeness, "closeness_centrality")

    return {
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
    }


def get_layout(G: nx.Graph, layout_type: str) -> dict:
    """Generate node positions based on layout type."""
    if layout_type == "geo":
        # Use exact coordinates from DB
        return {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    elif layout_type == "kamada_kawai":
        log.info("Calculating Kamada-Kawai layout (may take a moment)...")
        return nx.kamada_kawai_layout(G, weight="distance_m")
    elif layout_type == "spring":
        log.info("Calculating Spring layout...")
        return nx.spring_layout(G, weight="distance_m", seed=42)
    elif layout_type == "spectral":
        return nx.spectral_layout(G, weight="distance_m")
    elif layout_type == "circular":
        return nx.circular_layout(G)
    elif layout_type == "shell":
        return nx.shell_layout(G)
    elif layout_type == "spiral":
        return nx.spiral_layout(G)
    elif layout_type == "planar":
        try:
            return nx.planar_layout(G)
        except nx.NetworkXException as e:
            log.warning("Graph is not planar! Falling back to Kamada-Kawai layout: %s", e)
            return nx.kamada_kawai_layout(G, weight="distance_m")
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")


def plot_network(
    G: nx.Graph,
    pos: dict,
    metric_values: dict[str, float],
    metric_name: str,
    output_path: Path,
    title: str,
) -> None:
    """Generate and save a single network plot."""
    log.info("Plotting %s...", title)
    
    plt.figure(figsize=(16, 16))
    plt.style.use("dark_background")

    # Node sizes based on metric (scaled)
    # Normalize metric values to a reasonable node size range [50, 500]
    min_val = min(metric_values.values())
    max_val = max(metric_values.values())
    val_range = max_val - min_val if max_val > min_val else 1
    
    node_sizes = [
        50 + 450 * ((metric_values[n] - min_val) / val_range)
        for n in G.nodes()
    ]

    # Node colors based on metric
    node_colors = [metric_values[n] for n in G.nodes()]

    # Edge colors based on distance
    edge_weights = [d.get("distance_m", 1.0) for u, v, d in G.edges(data=True)]

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.4,
        edge_color=edge_weights,
        edge_cmap=plt.cm.viridis,
        width=1.5,
    )

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        alpha=0.9,
    )

    # Colorbars
    plt.colorbar(nodes, label=metric_name, shrink=0.5, pad=0.02)
    
    plt.title(title, fontsize=20, pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()
    log.info("Saved plot to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Transmilenio Network from DB")
    parser.add_argument(
        "--layout", 
        type=str, 
        default="kamada_kawai", 
        choices=["kamada_kawai", "spring", "spectral", "circular", "shell", "spiral", "planar"],
        help="Algorithm for abstract graph layouts"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/workflow/results/network",
        help="Directory to save the analysis outputs"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to parameters file (provided by UI runner, ignored here)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 1. Load data
    G_full = load_graph_from_db()
    
    # Split into connected components (largest first)
    components = sorted(nx.connected_components(G_full), key=len, reverse=True)
    log.info("Found %d connected component(s). Only plotting the largest one.", len(components))
    
    # Extract the largest component (this automatically drops isolated nodes and smaller motifs)
    G = G_full.subgraph(components[0]).copy()
    
    log.info("Processing main network (%d nodes, %d edges)...", G.number_of_nodes(), G.number_of_edges())
    
    # 2. Compute metrics
    metrics = compute_metrics(G)
    
    # 3. Generate Layouts
    pos_geo = get_layout(G, "geo")
    pos_abstract = get_layout(G, args.layout)
    
    # 4. Generate Plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4a. Geo layout - Degree
    plot_network(
        G, 
        pos=pos_geo, 
        metric_values=metrics["degree"], 
        metric_name="Degree Centrality",
        output_path=output_dir / "network_geo_degree.png",
        title="Transmilenio Network (Geographic) - Degree Centrality",
    )
    
    # 4b. Geo layout - Betweenness
    plot_network(
        G, 
        pos=pos_geo, 
        metric_values=metrics["betweenness"], 
        metric_name="Betweenness Centrality",
        output_path=output_dir / "network_geo_betweenness.png",
        title="Transmilenio Network (Geographic) - Betweenness Centrality",
    )
    
    # 4c. Geo layout - Closeness
    plot_network(
        G, 
        pos=pos_geo, 
        metric_values=metrics["closeness"], 
        metric_name="Closeness Centrality",
        output_path=output_dir / "network_geo_closeness.png",
        title="Transmilenio Network (Geographic) - Closeness Centrality",
    )
    
    # 4d. Abstract layout - Degree
    plot_network(
        G, 
        pos=pos_abstract, 
        metric_values=metrics["degree"], 
        metric_name="Degree Centrality",
        output_path=output_dir / f"network_abstract_{args.layout}_degree.png",
        title=f"Transmilenio Network ({args.layout} layout) - Degree Centrality",
    )
        
    # 4e. Abstract layout - Betweenness
    plot_network(
        G, 
        pos=pos_abstract, 
        metric_values=metrics["betweenness"], 
        metric_name="Betweenness Centrality",
        output_path=output_dir / f"network_abstract_{args.layout}_betweenness.png",
        title=f"Transmilenio Network ({args.layout} layout) - Betweenness Centrality",
    )
    
    # 4f. Abstract layout - Closeness
    plot_network(
        G, 
        pos=pos_abstract, 
        metric_values=metrics["closeness"], 
        metric_name="Closeness Centrality",
        output_path=output_dir / f"network_abstract_{args.layout}_closeness.png",
        title=f"Transmilenio Network ({args.layout} layout) - Closeness Centrality",
    )
        
    # 5. Export JSON
    with open(output_dir / "network_graph.json", "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f, indent=2, ensure_ascii=False)
        
    log.info("Analysis complete! All plots and JSON saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
