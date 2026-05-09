"""
routers/realtime.py

REST + WebSocket endpoints for the real-time simulation module.

REST
----
POST   /api/realtime/session          Create session (pre-generates events)
GET    /api/realtime/session/{id}     Session metadata
DELETE /api/realtime/session/{id}     Destroy session
GET    /api/realtime/sessions         List active sessions
GET    /api/realtime/dates            Dates with real data + day_type
GET    /api/realtime/model_status     Which models are ready for given station/date
GET    /api/realtime/stations         Stations available in the active pipeline

WebSocket
---------
WS  /ws/realtime/{session_id}
    Client sends: {"t": <seconds from window start>}
    Server pushes: {
        "model_events": {sc: [t, ...]},
        "real_events":  {sc: [t, ...]},
        "forecast":     {sc: {time_hours, model_raw, corrected}},
        "adaptation":   {sc: {method, ratios, ...}},
        "t":            <echo of clock time>
    }
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.realtime.engine   import create_session, ModelType
from src.realtime.session_store import store
from src.realtime.loaders  import (
    available_dates_with_day_type,
    check_model_availability,
    query_day_type,
    build_model_inventory,
)

RESULTS_BASE = Path("src/workflow/results")
DATA_BASE    = Path("src/workflow/data")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    station_codes:     list[str]
    day_type:          str
    model:             str                     # hawkes | lgcp_prior | lgcp_posterior | avg_profile
    count_type:        str   = "checkins"      # checkins | checkouts
    date_str:          str   = ""              # Optional YYYYMMDD
    adaptation_method: str   = "bayesian"      # bayesian | multiplicative | hawkes_kappa | trend
    clock_start_hhmm:  int   = 400             # e.g. 700 for 07:00
    speed:             float = 1.0
    lookahead_min:     float = 60.0
    seed:              int   = 42
    run_id:            str   = ""              # specific experiment folder


# ---------------------------------------------------------------------------
# Helper: load pipeline stations
# ---------------------------------------------------------------------------

def _get_pipeline_stations() -> list[dict]:
    pipeline_dirs = sorted(
        [d for d in DATA_BASE.iterdir() if d.is_dir() and d.name.startswith("pipeline_")],
        key=lambda d: d.name,
        reverse=True,
    ) if DATA_BASE.exists() else []

    for pd_ in pipeline_dirs:
        stations_file = pd_ / "sampled_stations.csv"
        if stations_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(stations_file, dtype={"code": str})
                if "code" in df.columns:
                    return df.rename(columns={"code": "station_code", "name": "station_name"}).to_dict("records")
            except Exception:
                continue
    return []


def _get_fitted_cutoff_date() -> Optional[str]:
    """Finds the cutoff_date from the most recent model fit metadata."""
    for model_dir in ["lgcp_twostage", "hawkes_fit"]:
        base = RESULTS_BASE / model_dir
        if not base.exists():
            continue
        meta_files = sorted(base.rglob("run_meta.json"), reverse=True)
        for mf in meta_files:
            try:
                import json
                with open(mf, "r") as f:
                    meta = json.load(f)
                    params = meta.get("params", {})
                    if "cutoff_date" in params:
                        return str(params["cutoff_date"]).replace("-", "")
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.get("/dates")
async def get_available_dates():
    """Return all dates that have real data in the DB, with day_type."""
    try:
        dates = available_dates_with_day_type()
        cutoff = _get_fitted_cutoff_date()
        return {"dates": dates, "cutoff_date": cutoff}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stations")
async def get_stations():
    """Return stations from the most recent pipeline."""
    return {"stations": _get_pipeline_stations()}


@router.get("/inventory")
async def get_inventory():
    """
    Returns full hierarchy of available fitted models:
    { "inventory": [ { "station_code": "...", "station_name": "...", "count_types": { "checkins": { "WD": ["hawkes", ...] }, ... } } ] }
    """
    stations = _get_pipeline_stations()
    inventory_map = build_model_inventory(RESULTS_BASE)
    
    result = []
    for st in stations:
        sc = st["station_code"]
        if sc in inventory_map:
            result.append({
                "station_code": sc,
                "station_name": st["station_name"],
                "count_types": inventory_map[sc]
            })
    return {"inventory": result}


@router.get("/model_status")
async def get_model_status(date_str: str, stations: str, count_type: str = "checkins"):
    """
    Query which models have fitted params for the given date + stations.
    `stations` is a comma-separated list of station codes.
    Returns {hawkes: bool, lgcp_prior: bool, lgcp_posterior: bool, avg_profile: bool}
    """
    station_list = [s.strip().zfill(5) for s in stations.split(",") if s.strip()]
    day_type = query_day_type(date_str) or "WD"
    try:
        status = check_model_availability(RESULTS_BASE, station_list, day_type, count_type)
        status["day_type"] = day_type
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session")
async def create_realtime_session(req: CreateSessionRequest):
    """
    Create a new real-time session: load model params, pre-generate events, load real data.
    This may take a few seconds for large models.
    """
    try:
        session = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: create_session(
                date_str          = req.date_str,
                day_type          = req.day_type,
                station_codes     = req.station_codes,
                model             = req.model,          # type: ignore[arg-type]
                count_type        = req.count_type,
                adaptation_method = req.adaptation_method,  # type: ignore[arg-type]
                clock_start_hhmm  = req.clock_start_hhmm,
                speed             = req.speed,
                lookahead_min     = req.lookahead_min,
                seed              = req.seed,
                run_id            = req.run_id,
            ),
        )
        store.put(session)

        return {
            "session_id":    session.session_id,
            "day_type":      session.day_type,
            "has_real_data": session.has_real_data,
            "station_codes": session.station_codes,
            "model":         session.model,
            "model_event_counts": {
                sc: int(len(evs)) for sc, evs in session.model_events.items()
            },
            "real_event_counts": {
                sc: int(len(evs)) for sc, evs in session.real_events.items()
            },
            "clock_start_sec": session.clock_start_sec,
        }
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    sess = store.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id":    sess.session_id,
        "date_str":      sess.date_str,
        "day_type":      sess.day_type,
        "model":         sess.model,
        "station_codes": sess.station_codes,
        "has_real_data": sess.has_real_data,
        "adaptation_method": sess.adaptation_method,
        "clock_start_sec":   sess.clock_start_sec,
        "speed":             sess.speed,
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    removed = store.delete(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@router.get("/sessions")
async def list_sessions():
    purged = store.purge_expired()
    return {"sessions": store.list_sessions(), "purged": purged}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

async def realtime_ws_endpoint(websocket: WebSocket, session_id: str):
    """
    Clock-driven WebSocket.
    Client sends: {"t": <float seconds from window start>}
    Server returns a batch of events + forecast + adaptation state.
    """
    await websocket.accept()

    sess = store.get(session_id)
    if sess is None:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    _prev_t: float = sess.clock_start_sec

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t_now = float(msg.get("t", _prev_t))
            t_prev = _prev_t
            _prev_t = t_now

            # Slice events
            events    = sess.get_events_in_window(t_prev, t_now)
            forecast  = sess.get_forecast(t_now)
            adaptation = sess.get_adaptation_state()

            await websocket.send_json({
                "t":            t_now,
                "model_events": events,
                "forecast":     forecast,
                "adaptation":   adaptation,
                "has_real_data": sess.has_real_data,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
