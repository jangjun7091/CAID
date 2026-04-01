"""
CAID REST API — FastAPI routes.

Endpoints:
  POST   /designs                          Submit a design brief, returns session_id.
  GET    /designs/{session_id}             Poll session status and results.
  GET    /designs/{session_id}/artifacts/{name}  Download a STEP or STL file.
  POST   /designs/{session_id}/refine      Append human feedback and trigger re-run.
  GET    /materials                        List available materials.
  GET    /processes                        List available manufacturing processes.
  GET    /health                           Liveness probe.

Design sessions run synchronously in a background thread so the POST /designs
endpoint returns immediately with a session_id. Callers poll GET /designs/{id}
until status is "complete" or "failed".
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.orchestrator import AgentOrchestrator
from core.session import DesignSession
from core.world_model import WorldModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CAID API",
    description="Computer AI Design — Agentic hardware design framework",
    version="0.2.0",
)

# ---------------------------------------------------------------------------
# In-memory session store (Phase 2: replace with Redis / DB in Phase 3)
# ---------------------------------------------------------------------------

SessionStatus = Literal["running", "complete", "failed"]

_sessions: dict[str, dict] = {}   # session_id -> {"status": ..., "session": DesignSession | None, "error": str | None}
_sessions_lock = threading.Lock()

_world_model = WorldModel()
_OUTPUT_DIR = Path(os.environ.get("CAID_OUTPUT_DIR", "output"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DesignRequest(BaseModel):
    brief: str
    max_iterations: int = 3


class DesignResponse(BaseModel):
    session_id: str
    status: SessionStatus
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: SessionStatus
    all_passed: Optional[bool]
    components: Optional[dict]
    error: Optional[str]


class RefineRequest(BaseModel):
    feedback: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}


@app.get("/materials")
def list_materials() -> dict:
    return {"materials": _world_model.list_materials()}


@app.get("/processes")
def list_processes() -> dict:
    return {"processes": _world_model.list_processes()}


@app.post("/designs", response_model=DesignResponse, status_code=202)
def submit_design(request: DesignRequest, background_tasks: BackgroundTasks) -> DesignResponse:
    """
    Submit a design brief. Returns a session_id immediately.
    The design loop runs in a background thread.
    """
    session_id = str(uuid.uuid4())

    with _sessions_lock:
        _sessions[session_id] = {"status": "running", "session": None, "error": None}

    background_tasks.add_task(_run_design, session_id, request)

    return DesignResponse(
        session_id=session_id,
        status="running",
        message=f"Design loop started. Poll GET /designs/{session_id} for status.",
    )


@app.get("/designs/{session_id}", response_model=SessionStatusResponse)
def get_session(session_id: str) -> SessionStatusResponse:
    """Poll the status of a design session."""
    entry = _get_session_or_404(session_id)

    session: Optional[DesignSession] = entry.get("session")
    components = None
    if session is not None:
        components = {
            name: {
                "passed": r.passed,
                "iterations": len(session.iterations_for.get(name, [])),
                "step_path": str(r.step_path) if r.step_path else None,
                "findings": [
                    {"severity": f.severity.value, "message": f.message}
                    for f in r.final_critique.findings
                ],
            }
            for name, r in session.results.items()
        }

    return SessionStatusResponse(
        session_id=session_id,
        status=entry["status"],
        all_passed=session.all_passed if session else None,
        components=components,
        error=entry.get("error"),
    )


@app.get("/designs/{session_id}/artifacts/{name}")
def download_artifact(session_id: str, name: str, fmt: str = "step") -> FileResponse:
    """
    Download a STEP or STL artifact for a completed design component.

    Args:
        name: Component name (e.g., "mounting_bracket").
        fmt: "step" or "stl" (default: "step").
    """
    entry = _get_session_or_404(session_id)
    session: Optional[DesignSession] = entry.get("session")

    if session is None or entry["status"] != "complete":
        raise HTTPException(status_code=409, detail="Session is not yet complete.")

    results = session.results
    if name not in results:
        raise HTTPException(status_code=404, detail=f"Component '{name}' not found.")

    result = results[name]
    path = result.step_path if fmt == "step" else result.stl_path

    if path is None or not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{fmt.upper()} file for '{name}' is not available.",
        )

    media_type = "application/step" if fmt == "step" else "model/stl"
    return FileResponse(path=str(path), media_type=media_type, filename=path.name)


@app.post("/designs/{session_id}/refine", response_model=DesignResponse, status_code=202)
def refine_session(
    session_id: str, request: RefineRequest, background_tasks: BackgroundTasks
) -> DesignResponse:
    """
    Append human feedback to a completed session and trigger a new design run
    with the feedback appended to the original brief.
    """
    entry = _get_session_or_404(session_id)
    session: Optional[DesignSession] = entry.get("session")

    if session is None:
        raise HTTPException(status_code=409, detail="Session has no completed run yet.")

    original_brief = session.brief
    refined_brief = f"{original_brief}\n\nAdditional requirements: {request.feedback}"

    new_session_id = str(uuid.uuid4())
    design_request = DesignRequest(brief=refined_brief)

    with _sessions_lock:
        _sessions[new_session_id] = {"status": "running", "session": None, "error": None}

    background_tasks.add_task(_run_design, new_session_id, design_request)

    return DesignResponse(
        session_id=new_session_id,
        status="running",
        message=f"Refinement started. Poll GET /designs/{new_session_id} for status.",
    )


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_design(session_id: str, request: DesignRequest) -> None:
    """Background task: run the full CAID orchestrator and store the result."""
    try:
        orchestrator = AgentOrchestrator.create(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=os.environ.get("CAID_MODEL", "claude-sonnet-4-6"),
            output_dir=_OUTPUT_DIR / session_id,
            max_iterations=request.max_iterations,
        )
        session = orchestrator.run(request.brief)

        with _sessions_lock:
            _sessions[session_id]["session"] = session
            _sessions[session_id]["status"] = "complete"

        logger.info("Session %s complete. all_passed=%s", session_id, session.all_passed)

    except Exception as exc:
        logger.exception("Session %s failed: %s", session_id, exc)
        with _sessions_lock:
            _sessions[session_id]["status"] = "failed"
            _sessions[session_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_session_or_404(session_id: str) -> dict:
    with _sessions_lock:
        entry = _sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return entry
