"""Download Transmilenio shapefiles from Datos Abiertos (ArcGIS Hub).

Downloads ZIP archives containing ESRI Shapefiles for stations, roads,
and operational connections, then extracts them to the appropriate
data directories.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

log = logging.getLogger(__name__)

# Default download URLs (ArcGIS Hub / Datos Abiertos Bogotá)
DEFAULT_URLS = {
    "stations": "https://hub.arcgis.com/api/v3/datasets/5365d814bbdd4062a59234eea7d70db7_2/downloads/data?format=shp&spatialRefId=3116&where=1%3D1",
    "roads": "https://hub.arcgis.com/api/v3/datasets/4f5282678c72406bb19f7fbf22886bbf_5/downloads/data?format=shp&spatialRefId=3116&where=1%3D1",
    "connections": "https://hub.arcgis.com/api/v3/datasets/68d51aada9f54e229237449dd0f8f8d9_4/downloads/data?format=shp&spatialRefId=3116&where=1%3D1",
}

DEFAULT_DIRS = {
    "stations": Path("data/geometry/stations"),
    "roads": Path("data/geometry/roads"),
    "connections": Path("data/geometry/connections"),
}


def _has_shapefiles(directory: Path) -> bool:
    """Check if directory already contains at least one .shp file."""
    if not directory.exists():
        return False
    return any(directory.glob("*.shp"))


def download_and_extract(
    url: str,
    target_dir: Path,
    name: str = "shapefile",
    force: bool = False,
) -> None:
    """Download a ZIP from *url* and extract to *target_dir*.
    
    Skips download if target_dir already contains .shp files,
    unless *force* is True.
    """
    if not force and _has_shapefiles(target_dir):
        log.info("  %s: shapefiles already exist in %s — skipping download", name, target_dir)
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    log.info("  %s: downloading from %s ...", name, url[:80])

    try:
        req = Request(url, headers={"User-Agent": "osltm-pipeline/1.0"})
        with urlopen(req, timeout=120) as resp:
            data = resp.read()

        log.info("  %s: downloaded %.1f KB, extracting...", name, len(data) / 1024)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(target_dir)

        log.info("  %s: extracted %d files to %s", name, len(zf.namelist()), target_dir)
    except Exception as e:
        log.error("  %s: Failed to download or extract shapefiles from %s: %s", name, url, str(e))
        raise RuntimeError(f"Failed to download {name} shapefiles: {e}") from e


def download_all_shapefiles(
    force: bool = False,
    urls: dict[str, str] | None = None,
    dirs: dict[str, Path] | None = None,
) -> None:
    """Download all three shapefile datasets.
    
    Parameters
    ----------
    force : bool
        Re-download even if files exist.
    urls : dict, optional
        Override default URLs. Keys: 'stations', 'roads', 'connections'.
    dirs : dict, optional
        Override default target directories.
    """
    urls = {**DEFAULT_URLS, **(urls or {})}
    dirs = {**DEFAULT_DIRS, **(dirs or {})}

    log.info("Downloading Transmilenio shapefiles from Datos Abiertos...")
    for key in ["stations", "roads", "connections"]:
        download_and_extract(
            url=urls[key],
            target_dir=dirs[key],
            name=key.capitalize(),
            force=force,
        )
    log.info("All shapefiles ready.")
