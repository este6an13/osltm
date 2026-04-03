from fastapi import APIRouter
from src.ui.backend.routers.analysis import run_script, ScriptRunRequest

router = APIRouter()

# Models are actually just scripts in the registry, 
# so we can just re-use the analysis runner by constructing the script_id

@router.post("/{model_name}/{step}/run")
async def run_model_step(model_name: str, step: str, req: ScriptRunRequest):
    # e.g. "lgcp", "step1_twostage" -> "models/lgcp/step1_twostage"
    script_id = f"models/{model_name}/{step}"
    return await run_script(script_id, req)
