import sys
import asyncio

# Windows requires ProactorEventLoop for subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import warnings
warnings.filterwarnings("ignore")


from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.ui.backend.routers import pipeline, analysis, models, results, status
from src.ui.backend.ws import websocket_endpoint

app = FastAPI(title="OSLTM Workflow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since it's a local UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router, prefix="/api/status", tags=["status"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(results.router, prefix="/api/results", tags=["results"])

app.add_api_websocket_route("/ws/runs/{run_id}", websocket_endpoint)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.ui.backend.main:app", host="127.0.0.1", port=8000, reload=True)
