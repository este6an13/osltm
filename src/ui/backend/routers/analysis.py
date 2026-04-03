from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from src.ui.backend.services.registry import SCRIPTS
from src.ui.backend.services.runner import runner
import pandas as pd
import os

router = APIRouter()

class ScriptRunRequest(BaseModel):
    params: Dict[str, Any]

@router.get("/")
def list_scripts():
    return SCRIPTS

@router.post("/{script_id:path}/run")
async def run_script(script_id: str, req: ScriptRunRequest):
    if script_id not in SCRIPTS:
        raise HTTPException(status_code=404, detail="Script not found")
        
    script_def = SCRIPTS[script_id]
    run_id = await runner.run_script(script_def.module, req.params)
    return {"run_id": run_id}

@router.get("/stations")
def get_stations():
    station_file = "d:/dequi/repositories/osltm/src/workflow/data/sampled_stations.csv"
    if os.path.exists(station_file):
        df = pd.read_csv(station_file)
        if 'code' in df.columns:
            df = df.rename(columns={'code': 'station_code', 'name': 'station_name'})
            df['station_code'] = df['station_code'].astype(str).str.zfill(5)
            # return as list of dicts
            return df.to_dict('records')
    return []
