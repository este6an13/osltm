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
    file_path = None
    if pipeline_id == "default" and experiment_id == "default":
        fallback_path = RESULTS_DIR / output_dir / filename
        if fallback_path.exists():
            file_path = fallback_path

    if file_path is None:
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
    file_path = None
    if pipeline_id == "default" and experiment_id == "default":
        fallback_path = RESULTS_DIR / output_dir / filename
        if fallback_path.exists():
            file_path = fallback_path

    if file_path is None:
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


@router.get("/service/headway_fitting/{pipeline_id}/{experiment_id}/fit")
def get_headway_fitting_results(pipeline_id: str, experiment_id: str, route: Optional[str] = None):
    """
    Load simulated_headways.csv and fitted_headway_report.json,
    generate histogram bins and smooth PDF curves for Gamma, Erlang,
    and Log-Normal fits. Returns them as a structured JSON object.
    Supports query parameter `route` to view multiple fitted routes.
    """
    import numpy as np
    import scipy.stats as stats
    
    # 1. Resolve output path
    base_path = None
    if pipeline_id != "default" and experiment_id != "default":
        try:
            base_path = _safe_path(RESULTS_DIR, "headway_fitting", pipeline_id, experiment_id)
        except HTTPException:
            pass
            
    if base_path is None or not base_path.exists():
        # Fallback to the canonical one if exists
        fallback = RESULTS_DIR / "headway_fitting"
        if fallback.exists() and (fallback / "simulated_headways.csv").exists():
            base_path = fallback
        else:
            raise HTTPException(status_code=404, detail="Headway fitting results not found")
            
    # Scan for available routes in this experiment folder
    available_routes = []
    if base_path.exists():
        for f in base_path.iterdir():
            if f.name.startswith("fitted_headway_report_") and f.name.endswith(".json"):
                r = f.name.replace("fitted_headway_report_", "").replace(".json", "")
                available_routes.append(r)
    available_routes = sorted(available_routes)

    # Determine which route to fetch
    if not route:
        if available_routes:
            route = available_routes[0]
        else:
            route = None

    if route:
        report_file = base_path / f"fitted_headway_report_{route}.json"
        samples_file = base_path / f"simulated_headways_{route}.csv"
    else:
        report_file = base_path / "fitted_headway_report.json"
        samples_file = base_path / "simulated_headways.csv"
    
    if not report_file.exists() or not samples_file.exists():
        raise HTTPException(status_code=404, detail=f"Required result files for route '{route}' not found")
        
    try:
        # Load report
        with open(report_file) as f:
            report = json.load(f)
            
        # Load samples
        df_samples = pd.read_csv(samples_file)
        H = df_samples["headway_minutes"].values
        
        # Calculate histogram: 30 bins
        counts, bin_edges = np.histogram(H, bins=30, density=True)
        x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        histogram = [
            {
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i+1]),
                "x": float(x_mid[i]),
                "density": float(counts[i])
            }
            for i in range(len(counts))
        ]
        
        # Generate smooth curves
        x_min = 0.1
        x_max = float(np.max(H) * 1.1)
        x_grid = np.linspace(x_min, x_max, 150)
        
        # Fits parameters
        fits = report["fits"]
        gam = fits["gamma"]["params"]
        erl = fits["erlang"]["params"]
        logn = fits["lognormal"]["params"]
        
        curves = []
        for x in x_grid:
            x_val = float(x)
            density_gam = float(stats.gamma.pdf(x_val, gam["shape"], scale=gam["scale"]))
            density_erl = float(stats.erlang.pdf(x_val, erl["shape_k"], scale=erl["scale"]))
            density_logn = float(stats.lognorm.pdf(x_val, logn["sigma"], scale=logn["scale"]))
            curves.append({
                "x": x_val,
                "gamma": density_gam,
                "erlang": density_erl,
                "lognormal": density_logn
            })
            
        return {
            "metadata": {
                "route_name": report.get("route_name", "B12"),
                "period": report.get("period", "peak"),
                "cv": report.get("cv", 0.25),
                "scheduled_mean": report.get("scheduled_mean", 5.0),
                "simulated_mean": report.get("simulated_mean", 5.0),
                "simulated_std": report.get("simulated_std", 1.25),
            },
            "fits": fits,
            "histogram": histogram,
            "curves": curves,
            "available_routes": available_routes,
            "selected_route": route or report.get("route_name", "B12")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile headway fits: {str(e)}")


@router.get("/service/headway_fitting/{pipeline_id}/{experiment_id}/traversal")
def get_traversal_simulation_results(pipeline_id: str, experiment_id: str):
    """
    Load and serve the traversal_simulation.json file for the given pipeline and experiment.
    Falls back to the root headway_fitting folder if not found.
    """
    file_path = None
    if pipeline_id != "default" and experiment_id != "default":
        try:
            file_path = _safe_path(RESULTS_DIR, "headway_fitting", pipeline_id, experiment_id, "traversal_simulation.json")
        except HTTPException:
            pass
            
    if file_path is None or not file_path.exists():
        fallback = RESULTS_DIR / "headway_fitting" / "traversal_simulation.json"
        if fallback.exists():
            file_path = fallback
        else:
            raise HTTPException(status_code=404, detail="Traversal simulation JSON not found")
            
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse traversal simulation JSON: {str(e)}")



