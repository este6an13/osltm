"""
Step 5: Generate Station Network

Downloads shapefiles from Datos Abiertos, builds the network graph,
and persists nodes and edges to the database.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.db.session_v2 import SessionLocal
from src.network.download_shapefiles import download_all_shapefiles, DEFAULT_DIRS
from src.network.generate_network import build_network, export_json, export_csv, plot_network, verify_network
from src.repo.v2.network.repository import NetworkRepository

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/geometry/output")


def run(params: dict[str, Any]) -> None:
    """Execute step 5: Generate network."""
    step_params = params.get("step5", {})
    force_redownload = step_params.get("force_redownload", False)
    force_regenerate = step_params.get("force_regenerate", False)
    generate_plot = step_params.get("generate_plot", True)

    urls = {}
    if step_params.get("stations_url"):
        urls["stations"] = step_params["stations_url"]
    if step_params.get("roads_url"):
        urls["roads"] = step_params["roads_url"]
    if step_params.get("connections_url"):
        urls["connections"] = step_params["connections_url"]

    # 1. Download Shapefiles
    log.info("Phase 1: Download Shapefiles")
    download_all_shapefiles(force=force_redownload, urls=urls if urls else None)

    # 2. Check DB
    log.info("Phase 2: Database Persistence")
    db = SessionLocal()
    try:
        repo = NetworkRepository(db)
        if repo.has_data() and not force_regenerate:
            log.info("Network data already exists in database. Use force_regenerate=True to overwrite.")
            return

        # 3. Build Network
        log.info("Phase 3: Generate Network")
        
        # We need to find the actual .shp files inside the extracted directories
        # download_shapefiles extracts ZIPs into DEFAULT_DIRS
        
        def find_shp(directory: Path) -> Path:
            shps = list(directory.glob("*.shp"))
            if not shps:
                raise FileNotFoundError(f"No .shp file found in {directory}")
            return shps[0]

        stations_shp = find_shp(DEFAULT_DIRS["stations"])
        roads_shp = find_shp(DEFAULT_DIRS["roads"])
        connections_shp = find_shp(DEFAULT_DIRS["connections"])

        nodes, edges = build_network(
            stations_shp=stations_shp,
            roads_shp=roads_shp,
            connections_shp=connections_shp
        )
        
        verify_network(nodes, edges)

        # 4. Export JSON/CSV/Plot
        log.info("Phase 4: Export artifacts")
        export_json(nodes, edges, OUTPUT_DIR / "network.json")
        export_csv(nodes, edges, OUTPUT_DIR)
        if generate_plot:
            plot_network(nodes, edges, OUTPUT_DIR / "network_map.png")

        # 5. Persist to DB
        log.info("Phase 5: DB Upsert")
        nodes_dict = []
        for n in nodes:
            d = asdict(n)
            d["station_id"] = d.pop("id")  # Rename id to station_id for DB
            nodes_dict.append(d)

        edges_dict = [asdict(e) for e in edges]

        repo.replace_all(nodes_dict, edges_dict)
        log.info("Successfully persisted %d nodes and %d edges to database.", len(nodes), len(edges))

    finally:
        db.close()

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    with open("src/workflow/params.json", "r") as f:
        p = json.load(f)
    run(p)
