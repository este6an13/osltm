import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import shutil

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


@router.delete("/{output_dir}/{pipeline_id}/{experiment_id}")
def delete_experiment(output_dir: str, pipeline_id: str, experiment_id: str):
    """Delete a specific experiment."""
    exp_path = _safe_path(RESULTS_DIR, output_dir, pipeline_id, experiment_id)
    if not exp_path.exists() or not exp_path.is_dir():
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    shutil.rmtree(exp_path)
    return {"status": "deleted", "experiment_id": experiment_id}


@router.get("/gravity_od/{pipeline_id}/{experiment_id}/matrix")
def get_gravity_od_matrix(pipeline_id: str, experiment_id: str):
    """
    Parse the estimated_od_probabilities.csv into a structured JSON format 
    suitable for smooth, interactive frontend animations.
    """
    # 1. Resolve file path
    file_path = None
    if pipeline_id != "default" and experiment_id != "default":
        try:
            file_path = _safe_path(RESULTS_DIR, "gravity_od", pipeline_id, experiment_id, "estimated_od_probabilities.csv")
        except HTTPException:
            pass
            
    if file_path is None or not file_path.exists():
        # Fallback to the canonical one if exists
        fallback = RESULTS_DIR / "gravity_od" / "estimated_od_probabilities.csv"
        if fallback.exists():
            file_path = fallback
        else:
            raise HTTPException(status_code=404, detail="Estimated OD probabilities CSV not found")
            
    # 2. Read and parse
    try:
        df = pd.read_csv(file_path)
        # Sort values to ensure alignment
        df["origin_code"] = df["origin_code"].astype(str).str.zfill(5)
        df["destination_code"] = df["destination_code"].astype(str).str.zfill(5)
        
        # Get sorted unique station codes
        station_codes = sorted(df["origin_code"].unique())
        code_to_idx = {code: idx for idx, code in enumerate(station_codes)}
        
        # Map codes to names
        name_map = {}
        for _, row in df.drop_duplicates("origin_code").iterrows():
            name_map[row["origin_code"]] = row["origin_name"]
        for _, row in df.drop_duplicates("destination_code").iterrows():
            name_map[row["destination_code"]] = row["destination_name"]
            
        station_names = [name_map.get(code, code) for code in station_codes]
        
        # Get time bins in chronological order
        time_bins = sorted(df["time_bin"].unique(), key=lambda c: int(c.replace("t_", "")))
        
        # Build matrices
        flows = {t: [[0.0] * len(station_codes) for _ in range(len(station_codes))] for t in time_bins}
        probs = {t: [[0.0] * len(station_codes) for _ in range(len(station_codes))] for t in time_bins}
        
        for _, row in df.iterrows():
            t = row["time_bin"]
            o = row["origin_code"]
            d = row["destination_code"]
            if o in code_to_idx and d in code_to_idx:
                oi = code_to_idx[o]
                di = code_to_idx[d]
                flows[t][oi][di] = float(row["estimated_flow"])
                probs[t][oi][di] = float(row["routing_probability"])
                
        return {
            "station_codes": station_codes,
            "station_names": station_names,
            "time_bins": time_bins,
            "flows": flows,
            "probabilities": probs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse OD CSV: {str(e)}")

