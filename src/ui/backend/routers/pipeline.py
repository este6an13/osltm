import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.ui.backend.services.runner import runner
import shutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["pipeline"])

PARAMS_PATH = Path("src/workflow/params.json")
DATA_BASE = Path("src/workflow/data")


class PipelineRunRequest(BaseModel):
    steps: List[int]
    params: Optional[Dict[str, Any]] = None


@router.get("/params")
async def get_params():
    """Return current params.json contents."""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f:
            return json.load(f)
    return {}


@router.post("/params")
async def save_params(params: Dict[str, Any]):
    """Overwrite params.json with new values."""
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    return {"status": "saved"}


@router.post("/run")
async def run_pipeline(request: PipelineRunRequest):
    """Start the data pipeline and return run_id + pipeline_id."""
    # Load current params.json and merge with any UI overrides
    base_params: Dict[str, Any] = {}
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f:
            base_params = json.load(f)
    if request.params:
        base_params.update(request.params)

    result = await runner.run_pipeline(steps=request.steps, params=base_params)
    return result


@router.get("/experiments")
async def list_experiments():
    """List all pipeline runs, newest first."""
    return runner.list_pipeline_experiments()


@router.get("/experiments/active")
async def get_active_experiment():
    """Return the most recent pipeline run."""
    active = runner.get_active_pipeline()
    if active is None:
        return {"pipeline_id": None, "message": "No pipeline runs found"}
    return active


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline and all its associated experiments."""
    if not pipeline_id.startswith("pipeline_"):
        raise HTTPException(status_code=400, detail="Invalid pipeline ID")
    
    # 1. Delete data directory
    pipeline_dir = DATA_BASE / pipeline_id
    if pipeline_dir.exists() and pipeline_dir.is_dir():
        shutil.rmtree(pipeline_dir)
        
    # 2. Delete all results directories for this pipeline
    results_base = Path("src/workflow/results")
    if results_base.exists():
        for output_dir in results_base.iterdir():
            if output_dir.is_dir():
                pipeline_res_dir = output_dir / pipeline_id
                if pipeline_res_dir.exists() and pipeline_res_dir.is_dir():
                    shutil.rmtree(pipeline_res_dir)
                    
    return {"status": "deleted", "pipeline_id": pipeline_id}
