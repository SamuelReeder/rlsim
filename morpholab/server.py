from __future__ import annotations

import atexit
from datetime import datetime, timezone
from functools import lru_cache
import io
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
import threading
import time
from typing import Literal
import uuid
import zipfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import BodyConfig, EnvironmentConfig, get_presets
from .environment import WalkerEnv, export_mjcf
from .tasks import TaskConfig
from .training import (
    TrainConfig, atomic_json, evaluate_policy, hardware_info, read_json,
    replay_context, replay_run,
)

ACTIVE = {"starting", "training", "stopping"}
ID_PATTERN = re.compile(r"^\d{8}T\d{6}-[a-f0-9]{8}$")


class RunManager:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process = None
        self.active_id = None

    def resolve(self, run_id=None):
        if run_id is None:
            run_id = self.active_id
            if run_id is None:
                dirs = sorted(self.root.glob("*/config.json"), reverse=True)
                if not dirs:
                    raise HTTPException(404, "No saved runs yet.")
                run_id = dirs[0].parent.name
        if not ID_PATTERN.fullmatch(run_id):
            raise HTTPException(400, "Invalid run id.")
        path = self.root / run_id
        if not (path / "config.json").is_file():
            raise HTTPException(404, "Run does not exist.")
        return path

    def status(self):
        with self.lock:
            try:
                path = self.resolve()
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                return {"status": "idle", "steps": 0, "total_steps": 0, "elapsed": 0,
                        "history": [], "metrics": {}, "checkpoint_available": False}
            state = read_json(path / "status.json")
            if state["status"] in ACTIVE:
                if self.process is None or self.process.poll() is not None:
                    state.update(status="failed",
                                 error="Worker exited before finishing. See worker.log in the run folder.")
                    atomic_json(path / "status.json", state)
                elif (path / "STOP").exists():
                    state["status"] = "stopping"
            return state

    def start(self, config: TrainConfig, resume_from=None):
        with self.lock:
            if self.status()["status"] in ACTIVE:
                raise HTTPException(409, "A training run is already active.")
            if config.task is not None:
                try:
                    validation_env = WalkerEnv(config.body, environment=config.environment, task=config.task)
                    validation_env.close()
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
            try:
                hardware = hardware_info(config.device)
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            path = self.root / run_id
            path.mkdir()
            atomic_json(path / "config.json", config.model_dump())
            (path / "body.xml").write_text(
                export_mjcf(config.body, config.environment), encoding="utf-8"
            )
            state = {
                "status": "starting", "run_id": run_id, "steps": 0,
                "total_steps": config.actual_steps, "elapsed": 0.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {}, "history": [], "frame": None, "error": None,
                "config": config.model_dump(), "checkpoint_available": False,
                "hardware": hardware,
            }
            if resume_from is not None:
                from .continuation import read_state
                saved = read_state(resume_from)
                for name in ["policy.zip", "normalize.pkl", "learning_state.json", "best_policy.zip", "best_normalize.pkl", "best.json"]:
                    if (resume_from / name).exists():
                        shutil.copy2(resume_from / name, path / name)
                atomic_json(path / "resume.json", {"parent_run_id": resume_from.name, "start_steps": saved["steps"]})
                previous = read_json(resume_from / "status.json")
                state.update(parent_run_id=resume_from.name, start_steps=saved["steps"], steps=saved["steps"],
                             total_steps=saved["steps"] + config.actual_steps, history=previous.get("history", []),
                             best_validation=previous.get("best_validation"), best_checkpoint_steps=previous.get("best_checkpoint_steps"))
            atomic_json(path / "status.json", state)
            try:
                with (path / "worker.log").open("w") as log:
                    self.process = subprocess.Popen(
                        [sys.executable, "-m", "morpholab.cli", "worker", str(path)],
                        stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                    )
            except Exception as exc:
                state.update(status="failed", error=str(exc))
                atomic_json(path / "status.json", state)
                raise HTTPException(500, "Could not launch training worker.") from exc
            self.active_id = run_id
            return state

    def resume(self, run_id, additional_steps):
        from .continuation import read_state
        with self.lock:
            path = self.resolve(run_id)
            try:
                read_state(path)
            except (OSError, ValueError) as exc:
                raise HTTPException(409, "This run lacks complete, matching continuation state. Older checkpoints support inference only.") from exc
            config = TrainConfig.model_validate({**read_json(path / "config.json"), "total_steps": additional_steps})
            return self.start(config, resume_from=path)

    def stop(self):
        with self.lock:
            state = self.status()
            if state["status"] in ACTIVE:
                (self.resolve() / "STOP").touch()
                state["status"] = "stopping"
            return state

    def close(self):
        if self.process and self.process.poll() is None:
            (self.resolve(self.active_id) / "STOP").touch()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()


@lru_cache
def cached_hardware():
    return hardware_info()


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body: BodyConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    task: TaskConfig | None = None


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    run_id: str | None = None
    body: BodyConfig | None = None
    environment: EnvironmentConfig | None = None
    target_speed: float | None = Field(default=None, ge=0.1, le=3.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    deterministic: Literal[True] = True
    checkpoint: Literal["best", "latest"] = "best"
    task: TaskConfig | None = None
    replay_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,96}$")


class StopReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    replay_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,96}$")


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    additional_steps: int = Field(default=65536, ge=4096, le=2_000_000)


class ReplayJobs:
    """Request-scoped cancellation, including stop arriving before replay starts."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}
        self.pending = {}

    def start(self, token):
        with self.lock:
            if token in self.active:
                raise HTTPException(409, "This inference request is already running.")
            event = threading.Event()
            stopped_at = self.pending.pop(token, None)
            if stopped_at is not None and time.monotonic() - stopped_at < 60:
                event.set()
            self.active[token] = event
            return event

    def stop(self, token):
        with self.lock:
            if token in self.active:
                self.active[token].set()
            else:
                now = time.monotonic()
                self.pending = {key: stamp for key, stamp in self.pending.items() if now - stamp < 60}
                if len(self.pending) >= 256:
                    self.pending.pop(next(iter(self.pending)))
                self.pending[token] = now

    def finish(self, token):
        with self.lock:
            self.active.pop(token, None)


def create_app(runs_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="RLsim", version="0.1.0")
    manager = RunManager(runs_dir or Path(os.environ.get("MORPHOLAB_RUNS", "runs")).resolve())
    app.state.manager = manager
    replay_jobs = ReplayJobs()
    app.state.replay_jobs = replay_jobs
    atexit.register(manager.close)
    from .rig_jobs import RigJobs
    from .rig_prepare import MAX_GLB_BYTES, inspect_glb, installation_status, RigPreparationError
    rig_jobs = RigJobs(manager.root.parent / '.rig-jobs')
    app.state.rig_jobs = rig_jobs
    atexit.register(rig_jobs.close)

    def rig_access(request):
        # The native helper is deliberately local-only. A hosted service needs
        # authentication and its own resource policy before enabling GPU jobs.
        from urllib.parse import urlsplit
        from ipaddress import ip_address
        client = request.client.host if request.client else ''
        try:
            address = ip_address(client)
            local_client = address.is_loopback or bool(getattr(address, 'ipv4_mapped', None) and address.ipv4_mapped.is_loopback)
        except ValueError:
            local_client = client == 'testclient' and request.url.hostname == 'testserver'
        if not local_client:
            raise HTTPException(403, 'Automatic rigging requires a local connection to RLsim.')
        if request.url.hostname not in {'localhost', '127.0.0.1', '::1', 'testserver'}:
            raise HTTPException(403, 'Automatic rigging is available from the local RLsim server only.')
        origin = request.headers.get('origin')
        if origin and (urlsplit(origin).netloc != request.url.netloc or urlsplit(origin).scheme != request.url.scheme):
            raise HTTPException(403, 'Open the local RLsim page to prepare a rig.')

    @app.get("/api/config")
    def config():
        return {
            "presets": get_presets(),
            "defaults": TrainConfig().model_dump(exclude={"body"}),
            "environment_defaults": EnvironmentConfig().model_dump(),
            "hardware": cached_hardware(),
        }

    @app.get("/api/rig-prepare/status")
    def rig_preparation_status():
        from .rig_prepare import installation_status
        # Read-only: this endpoint never imports or launches the external runtime.
        return installation_status()

    @app.post('/api/rig-prepare/jobs', status_code=202)
    async def start_rig_job(request: Request):
        rig_access(request)
        if not installation_status()['configured']:
            raise HTTPException(409, 'SkinTokens is not configured. Follow the setup guide, then restart the local server.')
        if manager.status()['status'] in ACTIVE:
            raise HTTPException(409, 'Stop native training before starting GPU rig preparation.')
        if request.headers.get('content-type', '').split(';')[0] != 'model/gltf-binary':
            raise HTTPException(415, 'Select one self-contained .glb model.')
        try:
            token = rig_jobs.reserve()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        path = rig_jobs.root / token / 'input.glb'
        try:
            size = 0
            with path.open('wb') as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_GLB_BYTES:
                        raise HTTPException(413, 'The model exceeds the 64 MB limit.')
                    handle.write(chunk)
            inspect_glb(path)
        except BaseException as exc:
            with rig_jobs.lock:
                rig_jobs.jobs[token].update(status='failed', error='Model upload failed.')
            path.unlink(missing_ok=True)
            if isinstance(exc, (ValueError, RigPreparationError)):
                raise HTTPException(422, str(exc)) from exc
            raise
        return rig_jobs.start(token)

    @app.get('/api/rig-prepare/jobs/{token}')
    def rig_job(token: str, request: Request):
        rig_access(request)
        try:
            return rig_jobs.snapshot(token)
        except KeyError as exc:
            raise HTTPException(404, 'Rig preparation was not found. The local server may have restarted.') from exc

    @app.post('/api/rig-prepare/jobs/{token}/cancel')
    def cancel_rig_job(token: str, request: Request):
        rig_job(token, request)
        return rig_jobs.stop(token)

    @app.get('/api/rig-prepare/jobs/{token}/model')
    def rig_job_model(token: str, request: Request):
        if rig_job(token, request)['status'] != 'completed':
            raise HTTPException(409, 'The rigged model is not ready.')
        return FileResponse(rig_jobs.root / token / 'rigged.glb', media_type='model/gltf-binary', filename='rigged.glb')

    @app.post("/api/preview")
    def preview(request: PreviewRequest | BodyConfig):
        if isinstance(request, BodyConfig):
            request = PreviewRequest(body=request)
        try:
            env = WalkerEnv(request.body, environment=request.environment, task=request.task)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            env.reset(seed=7)
            return {"body": request.body.model_dump(),
                    "environment": request.environment.model_dump(),
                    "frame": env.get_frame(),
                    "action_size": env.action_space.shape[0],
                    "observation_size": env.observation_space.shape[0],
                    "model_info": env.model_info()}
        finally:
            env.close()

    @app.post("/api/train")
    def train(config: TrainConfig):
        if any(job['status'] in {'uploading','checking','rigging','verifying','stopping'} for job in rig_jobs.jobs.values()):
            raise HTTPException(409, 'Finish or cancel rig preparation before starting native training.')
        return manager.start(config)

    @app.post("/api/resume")
    def resume(request: ResumeRequest):
        if any(job['status'] in {'uploading','checking','rigging','verifying','stopping'} for job in rig_jobs.jobs.values()):
            raise HTTPException(409, 'Finish or cancel rig preparation before continuing native training.')
        return manager.resume(request.run_id, request.additional_steps)

    @app.get("/api/status")
    def status():
        return manager.status()

    @app.post("/api/stop")
    def stop():
        return manager.stop()

    @app.get("/api/runs")
    def runs():
        manager.status()
        records = []
        for path in sorted(manager.root.glob("*/status.json"), reverse=True)[:100]:
            state = read_json(path)
            records.append({
                "id": path.parent.name, "status": state["status"],
                "steps": state["steps"], "created_at": state["created_at"],
                "name": state["config"]["body"]["name"],
                "checkpoint_available": state.get("checkpoint_available", False),
                "config": state["config"], "evaluation": state.get("evaluation"),
                "model_info": state.get("model_info"),
                "resumable": (path.parent / "learning_state.json").is_file(),
                "parent_run_id": state.get("parent_run_id"),
                **({"best": {"steps": state["best_checkpoint_steps"], "validation": state["best_validation"]}}
                   if state.get("best_validation") else {}),
            })
        return {"runs": records}

    @app.post("/api/stop-replay")
    def stop_replay(request: StopReplayRequest):
        replay_jobs.stop(request.replay_id)
        return {"stopping": True}

    @app.post("/api/replay")
    def replay(request: ReplayRequest):
        event = replay_jobs.start(request.replay_id) if request.replay_id else None
        try:
            if event is not None and event.is_set():
                return {"cancelled": True, "frames": []}
            return generate_replay(request, event.is_set if event is not None else None)
        finally:
            if event is not None:
                replay_jobs.finish(request.replay_id)

    def generate_replay(request: ReplayRequest, cancel=None):
        if manager.status()["status"] in ACTIVE:
            raise HTTPException(409, "Stop training before evaluating a saved policy.")
        if request.body is not None and request.run_id is None:
            config = TrainConfig(
                body=request.body,
                environment=request.environment or EnvironmentConfig(),
                target_speed=request.target_speed if request.target_speed is not None else 1.0,
                seed=request.seed if request.seed is not None else 7,
                task=request.task,
            )
            try:
                return {
                    **evaluate_policy(
                        config, capture=True,
                        seed=request.seed if request.seed is not None else 10007,
                        **({"cancel": cancel} if cancel else {}),
                    ),
                    "source": "untrained",
                }
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        if request.task is not None:
            raise HTTPException(422, "A saved policy uses its checkpoint task; task overrides are not supported.")
        path = manager.resolve(request.run_id)
        if not (path / "policy.zip").exists():
            raise HTTPException(409, "This run has no saved policy yet.")
        overrides = {
            key: getattr(request, key) for key in ("environment", "target_speed", "seed")
            if getattr(request, key) is not None
        }
        if not overrides and (request.checkpoint == "latest" or not (path / "best_policy.zip").exists()) and (path / "replay.json").exists():
            replay = read_json(path / "replay.json")
            config = TrainConfig.model_validate(read_json(path / "config.json"))
            return {
                **replay, **replay_context(config, replay["summary"].get("seed", 10007)),
                "source": "trained",
            }
        return replay_run(path, checkpoint=request.checkpoint, **overrides,
                          **({"cancel": cancel} if cancel else {}))

    @app.get("/api/checkpoint")
    def checkpoint(run_id: str | None = None):
        path = manager.resolve(run_id)
        if manager.status()["status"] in ACTIVE and path.name == manager.active_id:
            raise HTTPException(409, "Stop training before downloading this checkpoint.")
        if not (path / "policy.zip").exists():
            raise HTTPException(404, "No checkpoint is available.")
        # A policy needs its observation-normalization statistics and body config.
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ["policy.zip", "normalize.pkl", "config.json", "body.xml",
                         "evaluation.json", "baseline.json", "status.json",
                         "best_policy.zip", "best_normalize.pkl", "best.json", "learning_state.json", "resume.json"]:
                source = path / name
                if source.exists():
                    archive.write(source, name)
        return Response(buffer.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="rlsim-{path.name}.zip"'})

    @app.get("/api/body")
    def body(run_id: str | None = None):
        return FileResponse(manager.resolve(run_id) / "body.xml",
                            media_type="application/xml", filename="body.xml")

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
    return app

