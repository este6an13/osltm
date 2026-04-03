from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import os
from src.ui.backend.services.runner import runner

router = APIRouter()

PARAMS_FILE = "d:/dequi/repositories/osltm/src/workflow/params.json"

class PipelineRunRequest(BaseModel):
    steps: List[int]
    params: Dict[str, Any]

@router.get("/params")
def get_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "r") as f:
            return json.load(f)
    return {}

@router.put("/params")
def update_params(params: Dict[str, Any]):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)
    return {"status": "ok"}

@router.post("/run")
async def run_pipeline(req: PipelineRunRequest):
    # Optionally update params first
    if req.params:
        with open(PARAMS_FILE, "w") as f:
            json.dump(req.params, f, indent=4)
            
    run_id = await runner.run_pipeline(req.steps)
    return {"run_id": run_id}

@router.get("/runs/{run_id}")
def get_run_status(run_id: str):
    if run_id in runner.runs:
        return runner.runs[run_id]
    return {"error": "not found"}
