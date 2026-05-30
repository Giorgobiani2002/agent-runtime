"""
HTTP wrapper for the declario browser-agent runtime.

main.py is a CLI today; agent-backend used to invoke it via subprocess
on the same host (`spawn(PYTHON_BIN, ['main.py', '--bulk-run-id', X])`).
That model doesn't work on Railway because agent-backend runs in a
Node container with no Python or Chromium. This wrapper:

  - exposes `POST /run` which spawns a local `python main.py
    --bulk-run-id <id>` subprocess and returns immediately (HTTP 202);
  - lets agent-backend keep its existing pull-based worker pattern
    (claim row -> run -> report result), just over HTTP instead of
    fork-on-same-host;
  - tracks live worker pids per run so /run is idempotent and /workers
    can show what's executing.

Run in container as:
    uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}

Reachable inside Railway as http://agent-runtime.railway.internal:<port>.
Not exposed publicly — agent-backend is the only caller.
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MAIN_PY = str(ROOT / "main.py")
PYTHON_BIN = sys.executable

# Shared secret with agent-backend so this service can't be triggered
# by a stranger who lands on the internal hostname by accident.
EXPECTED_SECRET = os.environ.get("AI_INTERNAL_SECRET", "")

DEFAULT_CONCURRENCY = max(1, int(os.environ.get("BULK_WORKER_CONCURRENCY", "1")))

app = FastAPI(title="declario agent-runtime")

# run_id -> [pids]. Concurrency cap is per agent-runtime instance.
LIVE_WORKERS: Dict[str, List[int]] = {}


def _require_secret(provided: Optional[str]) -> None:
    if not EXPECTED_SECRET:
        # If no secret is configured, refuse — internal call MUST be authed.
        raise HTTPException(status_code=503, detail="AI_INTERNAL_SECRET not configured")
    if provided != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid X-Internal-Secret")


class RunRequest(BaseModel):
    bulk_run_id: str = Field(..., min_length=1)
    company_id: Optional[str] = None
    concurrency: Optional[int] = Field(None, ge=1, le=8)


class RunResponse(BaseModel):
    bulk_run_id: str
    spawned_pids: List[int]
    company_id: Optional[str] = None


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "agent-runtime",
        "python": sys.version.split()[0],
        "live_runs": list(LIVE_WORKERS.keys()),
        "live_workers": sum(len(v) for v in LIVE_WORKERS.values()),
    }


@app.post("/run", response_model=RunResponse, status_code=202)
async def run(
    payload: RunRequest,
    x_internal_secret: Optional[str] = Header(default=None, alias="X-Internal-Secret"),
):
    _require_secret(x_internal_secret)

    concurrency = payload.concurrency or DEFAULT_CONCURRENCY
    pids: List[int] = []

    env = os.environ.copy()
    if payload.company_id:
        env["AGENT_COMPANY_ID"] = payload.company_id
    # On Railway we force headless so we don't depend on a virtual display.
    env.setdefault("AGENT_HEADLESS", "true")

    for _ in range(concurrency):
        proc = await asyncio.create_subprocess_exec(
            PYTHON_BIN,
            MAIN_PY,
            "--bulk-run-id",
            payload.bulk_run_id,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(ROOT),
        )
        pids.append(proc.pid)
        # Detach: we don't await the subprocess. agent-backend already
        # tracks state via bulk_run_rows + recordHeartbeat — we don't
        # need to also track here. Reaping happens via os when the
        # process exits; on container exit Railway nukes everything.

    LIVE_WORKERS.setdefault(payload.bulk_run_id, []).extend(pids)
    return RunResponse(
        bulk_run_id=payload.bulk_run_id,
        spawned_pids=pids,
        company_id=payload.company_id,
    )


@app.post("/stop", status_code=200)
async def stop(
    payload: RunRequest,
    x_internal_secret: Optional[str] = Header(default=None, alias="X-Internal-Secret"),
):
    """Best-effort kill of workers for a run (e.g. user cancelled it)."""
    _require_secret(x_internal_secret)
    pids = LIVE_WORKERS.pop(payload.bulk_run_id, [])
    killed = 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass
    return {"bulk_run_id": payload.bulk_run_id, "killed": killed}


@app.get("/workers")
def workers(
    x_internal_secret: Optional[str] = Header(default=None, alias="X-Internal-Secret"),
):
    _require_secret(x_internal_secret)
    return {"runs": {k: v for k, v in LIVE_WORKERS.items()}}
