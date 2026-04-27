from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import pandas as pd

from src.ui.backend.services.registry import SCRIPTS
from src.ui.backend.services.runner import runner

router = APIRouter()

DATA_BASE = Path("src/workflow/data")
RESULTS_BASE = Path("src/workflow/results")


class ScriptRunRequest(BaseModel):
    params: Dict[str, Any]
    pipeline_id: Optional[str] = None   # sampling context
    exp_id: Optional[str] = None        # upstream experiment to inherit (downstream scripts)


@router.get("/")
def list_scripts():
    return SCRIPTS


@router.get("/stations")
def get_stations(pipeline_id: Optional[str] = Query(default=None)):
    """
    Return sampled stations. If pipeline_id is given, load the versioned snapshot
    for that pipeline run; fall back to the canonical file if not found.
    """
    def _load(path: Path):
        df = pd.read_csv(path)
        if "code" in df.columns:
            df = df.rename(columns={"code": "station_code", "name": "station_name"})
        df["station_code"] = df["station_code"].astype(str).str.zfill(5)
        return df.to_dict("records")

    if pipeline_id:
        versioned = DATA_BASE / pipeline_id / "sampled_stations.csv"
        if versioned.exists():
            return _load(versioned)

    canonical = DATA_BASE / "sampled_stations.csv"
    if canonical.exists():
        return _load(canonical)
    return []


@router.get("/upstream/{output_dir}")
def list_upstream_experiments(
    output_dir: str,
    pipeline_id: Optional[str] = Query(default=None),
):
    """
    List experiment runs inside a given output_dir (the depends_on dir).
    Experiments are stored under pipeline subdirectories.
    Optionally filter to those matching a specific pipeline_id.
    Returns experiments sorted newest-first with their run_meta.json contents.
    """
    dir_path = RESULTS_BASE / output_dir
    if not dir_path.exists():
        return []

    # Determine which pipeline dirs to scan
    if pipeline_id:
        pipeline_dirs = [dir_path / pipeline_id]
    else:
        pipeline_dirs = sorted(
            [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith("pipeline_")],
            reverse=True,
        )

    experiments = []
    for pd_ in pipeline_dirs:
        if not pd_.exists() or not pd_.is_dir():
            continue
        for entry in sorted(pd_.iterdir(), reverse=True):
            if not (entry.is_dir() and entry.name.startswith("exp_")):
                continue
            meta: Dict[str, Any] = {
                "experiment_id": entry.name,
                "pipeline_id": pd_.name,
            }
            meta_file = entry / "run_meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta.update(json.load(f))
                except Exception:
                    pass
            experiments.append(meta)

    return experiments


@router.post("/{script_id:path}/run")
async def run_script(script_id: str, req: ScriptRunRequest):
    if script_id not in SCRIPTS:
        raise HTTPException(status_code=404, detail="Script not found")

    script_def = SCRIPTS[script_id]
    result = await runner.run_script(
        module=script_def.module,
        script_key=script_id,
        output_dir=script_def.output_dir,
        params=req.params,
        pipeline_id=req.pipeline_id,
        exp_id=req.exp_id,
        depends_on=script_def.depends_on,
        input_arg=script_def.input_arg,
    )
    return result  # { run_id, experiment_id, output_subdir }
