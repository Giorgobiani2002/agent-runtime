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
import json
import os
import signal
import sys
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# Sentry — must init BEFORE FastAPI app is created so middleware wires up.
# No-op when SENTRY_DSN is unset.
SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=os.environ.get("RAILWAY_ENVIRONMENT_NAME", os.environ.get("ENV", "development")),
        release=os.environ.get("RAILWAY_DEPLOYMENT_ID"),
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        integrations=[FastApiIntegration()],
        # Don't capture request bodies — they contain tenant data (login
        # credentials, tax payloads) we never want in the error tracker.
        send_default_pii=False,
    )
    print(f"[sentry] initialised (env={os.environ.get('RAILWAY_ENVIRONMENT_NAME', '?')})")
else:
    print("[sentry] SENTRY_DSN not set — error reporting disabled")

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


class TaskRequest(BaseModel):
    """Autonomous (free-mode) browser-agent invocation. No pre-recorded
    playbook required — browser-use's LLM loop drives Chromium step by
    step from the task description + data.

    Used as the fallback path when agent-backend's dispatch lookup
    finds no playbook with the requested key.
    """

    task: str = Field(..., min_length=10)
    data: Dict[str, Any] = Field(default_factory=dict)
    company_id: Optional[str] = None
    # Optional correlation id so agent-backend can match the worker
    # output back to a declaration / approval queue row. We don't
    # interpret it here — just forward via env so main.py picks it up.
    correlation_id: Optional[str] = None
    max_steps: Optional[int] = Field(None, ge=1, le=200)
    safety_mode: Optional[str] = Field(default="halt-on-dangerous")


class TaskResponse(BaseModel):
    correlation_id: str
    spawned_pids: List[int]


@app.post("/run-task", response_model=TaskResponse, status_code=202)
async def run_task(
    payload: TaskRequest,
    x_internal_secret: Optional[str] = Header(default=None, alias="X-Internal-Secret"),
):
    """Spawn a worker in autonomous browser-use mode (no playbook).

    main.py supports --task / --data CLI args; we just hand those off.
    The worker logs to stdout; agent-backend correlates results via
    the correlation_id we stash in the AGENT_CORRELATION_ID env var.
    """
    _require_secret(x_internal_secret)

    correlation_id = payload.correlation_id or f"task-{_uuid.uuid4()}"

    env = os.environ.copy()
    if payload.company_id:
        env["AGENT_COMPANY_ID"] = payload.company_id
    env["AGENT_CORRELATION_ID"] = correlation_id
    env.setdefault("AGENT_HEADLESS", "true")

    args: List[str] = [
        PYTHON_BIN,
        MAIN_PY,
        "--task",
        payload.task,
        "--data",
        json.dumps(payload.data),
    ]
    if payload.max_steps:
        args.extend(["--max-steps", str(payload.max_steps)])
    if payload.safety_mode:
        args.extend(["--safety-mode", payload.safety_mode])

    proc = await asyncio.create_subprocess_exec(
        *args,
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(ROOT),
    )
    LIVE_WORKERS.setdefault(correlation_id, []).append(proc.pid)
    return TaskResponse(correlation_id=correlation_id, spawned_pids=[proc.pid])


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
