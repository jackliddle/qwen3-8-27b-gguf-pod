"""Image generation API for comfyui deployments.

- POST /v1/images/generations, /v1/images/edits: OpenAI-compatible, synchronous.
- POST /api/images/jobs + GET /api/images/jobs/{id}: async submit/poll, for
  generations that outlast RunPod's ~100s proxy timeout.
- GET /api/images/files/{subfolder}/{name}: result files. Unauthenticated so
  OpenAI-style `url` results work in browsers/clients; names carry a random
  128-bit job id, so they aren't guessable.
"""

import asyncio
import base64
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .auth import require_key
from .comfy_client import ComfyClient, ComfyError, output_images
from .comfy_workflow import build, set_output_prefix
from .config import get_settings
from .deployments import Deployment, DeploymentManager

SYNC_TIMEOUT = 900
JOB_TIMEOUT = 3600
MAX_JOBS = 500


class LoraUse(BaseModel):
    name: str
    strength: float | None = None


class ImageRequest(BaseModel):
    model: str
    prompt: str
    negative_prompt: str | None = None
    n: int = Field(1, ge=1, le=8)
    size: str | None = None  # "WxH", OpenAI style
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    loras: list[LoraUse] = []
    # Overrides for the deployment's recipe params (e.g. {"fast": true}).
    params: dict[str, Any] = {}
    response_format: Literal["b64_json", "url"] = "b64_json"
    # Async jobs only: input image for img2img recipes.
    image_b64: str | None = None


@dataclass
class Job:
    id: str
    model: str
    status: Literal["queued", "running", "done", "error"] = "queued"
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    seed: int | None = None
    images: list[dict] = field(default_factory=list)  # {filename, subfolder, type}
    task: asyncio.Task | None = None

    def to_dict(self, base: str) -> dict:
        return {
            "job_id": self.id,
            "model": self.model,
            "status": self.status,
            "error": self.error,
            "seed": self.seed,
            "created": self.created,
            "seconds": round((self.finished or time.time()) - self.created, 2),
            "images": [{"url": file_url(base, img)} for img in self.images],
        }


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def add(self, job: Job) -> None:
        self.jobs[job.id] = job
        if len(self.jobs) > MAX_JOBS:
            for jid in [j.id for j in self.jobs.values() if j.status in ("done", "error")][: len(self.jobs) - MAX_JOBS]:
                del self.jobs[jid]


def file_url(base: str, img: dict) -> str:
    sub = img.get("subfolder") or "_"
    return f"{base}/api/images/files/{sub}/{img['filename']}"


def _public_base(request: Request) -> str:
    return get_settings().public_base_url(str(request.base_url))


def _err(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error", "code": code}})


def resolve_deployment(manager: DeploymentManager, model: str) -> Deployment:
    for d in manager.deployments.values():
        if d.recipe.served_name == model and d.recipe.kind == "image":
            if d.status != "ready" or d.upstream is None:
                raise HTTPException(409, f"image model {model!r} is {d.status}, not ready")
            return d
    ready = [d.recipe.served_name for d in manager.deployments.values() if d.recipe.kind == "image" and d.status == "ready"]
    raise HTTPException(404, f"image model {model!r} is not deployed; ready image models: {ready}")


def _size(req: ImageRequest) -> tuple[int | None, int | None]:
    w, h = req.width, req.height
    if req.size and req.size != "auto":
        try:
            w, h = (int(x) for x in req.size.lower().split("x"))
        except ValueError:
            raise HTTPException(400, f"size must be WIDTHxHEIGHT, got {req.size!r}") from None
    # Latent models want multiples of 16.
    return (None if w is None else max(64, round(w / 16) * 16), None if h is None else max(64, round(h / 16) * 16))


def build_values(dep: Deployment, req: ImageRequest) -> tuple[dict[str, Any], list[tuple[str, float]]]:
    """Request + deployment params → (workflow values, LoRA chain)."""
    recipe = dep.recipe
    try:
        overrides = {k: recipe.params[k].coerce(v) for k, v in req.params.items() if k != "loras"}
    except KeyError as e:
        raise HTTPException(400, f"unknown param {e.args[0]!r} for {recipe.id}") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None

    available = set(dep.params.get("loras") or [])
    loras: list[tuple[str, float]] = []
    triggers: list[str] = []
    for lu in req.loras:
        if lu.name not in recipe.loras:
            raise HTTPException(400, f"unknown LoRA {lu.name!r}; catalog: {sorted(recipe.loras)}")
        if lu.name not in available:
            raise HTTPException(400, f"LoRA {lu.name!r} wasn't selected at deploy time; deployed: {sorted(available)}")
        spec = recipe.loras[lu.name]
        loras.append((lu.name, spec.strength if lu.strength is None else lu.strength))
        if spec.trigger:
            triggers.append(spec.trigger)

    width, height = _size(req)
    prompt = ", ".join([req.prompt, *triggers]) if triggers else req.prompt
    values = {
        **dep.params,
        **overrides,
        "prompt": prompt,
        "negative_prompt": req.negative_prompt,
        "width": width,
        "height": height,
        "seed": req.seed if req.seed is not None else random.randint(0, 2**31 - 1),
        "steps": req.steps,
        "cfg": req.cfg,
    }
    return values, loras


async def run_job(job: Job, dep: Deployment, req: ImageRequest, image: bytes | None) -> None:
    client = ComfyClient(dep.upstream.base_url)  # type: ignore[union-attr]
    try:
        values, loras = build_values(dep, req)
        job.seed = values["seed"]
        if image is not None:
            values["image"] = await client.upload(image, f"mp-{job.id}.png")
        job.status = "running"
        batch = "batch_size" in dep.recipe.inputs
        runs = 1 if batch else req.n
        if batch:
            values["batch_size"] = req.n
        for i in range(runs):
            wf = build(dep.recipe, {**values, "seed": values["seed"] + i}, loras)
            wf = set_output_prefix(wf, f"mp/{job.id}_{i}")
            prompt_id = await client.submit(wf)
            entry = await client.wait(prompt_id, timeout=JOB_TIMEOUT)
            job.images += output_images(entry)
        if not job.images:
            raise ComfyError("workflow finished without saving any image")
        job.status = "done"
    except HTTPException as e:
        job.status, job.error = "error", str(e.detail)
    except Exception as e:
        job.status, job.error = "error", str(e)
    finally:
        job.finished = time.time()


def start_job(request: Request, req: ImageRequest, image: bytes | None) -> Job:
    dep = resolve_deployment(request.app.state.manager, req.model)
    if image is not None and "img2img" not in dep.recipe.capabilities:
        raise HTTPException(400, f"{req.model} doesn't take an input image (not an img2img recipe)")
    if image is None and "txt2img" not in dep.recipe.capabilities:
        raise HTTPException(400, f"{req.model} needs an input image (use /v1/images/edits or image_b64)")
    job = Job(id=uuid.uuid4().hex, model=req.model)
    request.app.state.jobs.add(job)
    job.task = asyncio.create_task(run_job(job, dep, req, image))
    return job


async def _openai_response(request: Request, job: Job, fmt: str) -> JSONResponse:
    assert job.task is not None
    try:
        await asyncio.wait_for(asyncio.shield(job.task), timeout=SYNC_TIMEOUT)
    except TimeoutError:
        return _err(504, f"still running after {SYNC_TIMEOUT}s; poll /api/images/jobs/{job.id}", "timeout")
    if job.status != "done":
        return _err(500, job.error or "generation failed", "generation_failed")
    dep = resolve_deployment(request.app.state.manager, job.model)
    client = ComfyClient(dep.upstream.base_url)  # type: ignore[union-attr]
    base = _public_base(request)
    data = []
    for img in job.images:
        if fmt == "url":
            data.append({"url": file_url(base, img)})
        else:
            data.append({"b64_json": base64.b64encode(await client.view(img)).decode()})
    return JSONResponse({"created": int(job.created), "data": data, "seed": job.seed, "job_id": job.id})


# --- routes ---------------------------------------------------------------------

v1 = APIRouter(prefix="/v1/images", dependencies=[Depends(require_key)])
api = APIRouter(prefix="/api/images")
authed = APIRouter(dependencies=[Depends(require_key)])


@v1.post("/generations")
async def generations(req: ImageRequest, request: Request):
    if req.image_b64:
        raise HTTPException(400, "use /v1/images/edits for input images")
    job = start_job(request, req, None)
    return await _openai_response(request, job, req.response_format)


@v1.post("/edits")
async def edits(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form(...),
    n: int = Form(1),
    size: str | None = Form(None),
    response_format: Literal["b64_json", "url"] = Form("b64_json"),
    negative_prompt: str | None = Form(None),
    seed: int | None = Form(None),
    steps: int | None = Form(None),
    cfg: float | None = Form(None),
):
    req = ImageRequest(
        model=model, prompt=prompt, n=n, size=size, response_format=response_format,
        negative_prompt=negative_prompt, seed=seed, steps=steps, cfg=cfg,
    )
    job = start_job(request, req, await image.read())
    return await _openai_response(request, job, response_format)


@authed.post("/jobs")
async def create_job(req: ImageRequest, request: Request) -> dict:
    image = None
    if req.image_b64:
        try:
            image = base64.b64decode(req.image_b64.split(",")[-1])  # tolerate data: URLs
        except ValueError:
            raise HTTPException(400, "image_b64 is not valid base64") from None
    job = start_job(request, req, image)
    return job.to_dict(_public_base(request))


@authed.get("/jobs")
async def list_jobs(request: Request, limit: int = 50) -> list[dict]:
    base = _public_base(request)
    jobs = sorted(request.app.state.jobs.jobs.values(), key=lambda j: j.created, reverse=True)
    return [j.to_dict(base) for j in jobs[:limit]]


@authed.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict:
    job = request.app.state.jobs.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job.to_dict(_public_base(request))


@api.get("/files/{subfolder}/{filename}")
async def get_file(subfolder: str, filename: str) -> FileResponse:
    out = (get_settings().comfy_root / "output").resolve()
    path = (out / ("" if subfolder == "_" else subfolder) / filename).resolve()
    if not path.is_relative_to(out) or not path.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(path)


api.include_router(authed)
