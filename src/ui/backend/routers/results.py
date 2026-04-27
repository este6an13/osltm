import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

RESULTS_DIR = Path("d:/dequi/repositories/osltm/src/workflow/results")


def _safe_path(base: Path, *parts: str) -> Path:
    """Resolve path and verify it stays inside base (path traversal guard)."""
    resolved = (base / Path(*parts)).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    return resolved


@router.get("/")
def list_result_directories():
    """List top-level output directories (e.g. clustering_results, fpca_results)."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(d for d in os.listdir(RESULTS_DIR) if (RESULTS_DIR / d).is_dir())


@router.get("/{output_dir}")
def list_pipelines_and_experiments(output_dir: str):
    """
    List pipeline subdirectories inside an output_dir, each with its experiments.
    Returns a flat list of experiments annotated with their pipeline_id.
    """
    dir_path = _safe_path(RESULTS_DIR, output_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=404, detail="Output dir not found")

    experiments = []
    for pipeline_entry in sorted(dir_path.iterdir(), reverse=True):
        if not (pipeline_entry.is_dir() and pipeline_entry.name.startswith("pipeline_")):
            continue
        for exp_entry in sorted(pipeline_entry.iterdir(), reverse=True):
            if not (exp_entry.is_dir() and exp_entry.name.startswith("exp_")):
                continue
            meta: Dict[str, Any] = {
                "experiment_id": exp_entry.name,
                "pipeline_id": pipeline_entry.name,
            }
            meta_file = exp_entry / "run_meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta.update(json.load(f))
                except Exception:
                    pass
            experiments.append(meta)

    return experiments


@router.get("/{output_dir}/{pipeline_id}/{experiment_id}")
def list_experiment_files(output_dir: str, pipeline_id: str, experiment_id: str):
    """List files inside a specific experiment run (excluding run_meta.json)."""
    exp_path = _safe_path(RESULTS_DIR, output_dir, pipeline_id, experiment_id)
    if not exp_path.exists() or not exp_path.is_dir():
        raise HTTPException(status_code=404, detail="Experiment not found")

    files = sorted(
        f.name for f in exp_path.iterdir()
        if f.is_file() and f.name != "run_meta.json"
    )
    return {"experiment_id": experiment_id, "pipeline_id": pipeline_id, "files": files}


@router.get("/{output_dir}/{pipeline_id}/{experiment_id}/{filename}/view")
def view_file(output_dir: str, pipeline_id: str, experiment_id: str, filename: str):
    """Preview a file — returns JSON for CSV, binary for images."""
    file_path = _safe_path(RESULTS_DIR, output_dir, pipeline_id, experiment_id, filename)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(file_path)
            df = df.astype(object).replace([float("inf"), float("-inf")], None).where(pd.notnull(df), None)
            return {"columns": df.columns.tolist(), "data": df.to_dict("records")}
        except pd.errors.EmptyDataError:
            return {"columns": [], "data": []}
    else:
        return FileResponse(str(file_path))


@router.get("/{output_dir}/{pipeline_id}/{experiment_id}/{filename}/download")
def download_file(output_dir: str, pipeline_id: str, experiment_id: str, filename: str):
    """Download a result file."""
    file_path = _safe_path(RESULTS_DIR, output_dir, pipeline_id, experiment_id, filename)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=filename)
