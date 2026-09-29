import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import gateway
from .auth import require_key
from .config import Settings, get_settings
from .deployments import ACTIVE, DeployError, DeploymentManager
from .recipes import RecipeStore
from .system import disk_stats, gpu_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEV_RECIPES = Path(__file__).resolve().parents[1] / "dev-recipes"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)
    app.state.recipes = RecipeStore(s.recipes_dir, s.local_recipes_dir, DEV_RECIPES if s.mp_dev else None)
    app.state.manager = DeploymentManager(s)
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(10, read=None, write=120, pool=None))
    # Don't block startup on GitHub; baked-in recipes are usable immediately.
    refresh = asyncio.create_task(app.state.recipes.refresh_remote(s.recipes_repo, s.recipes_ref, s.github_token))
    yield
    refresh.cancel()
    await app.state.manager.shutdown()
    await app.state.http.aclose()


app = FastAPI(title="model-pod supervisor", lifespan=lifespan)
app.include_router(gateway.router)
api = APIRouter(prefix="/api")


@app.exception_handler(DeployError)
async def _deploy_error(_req: Request, exc: DeployError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


def _public_base(request: Request) -> str:
    return get_settings().public_base_url(str(request.base_url))


@api.get("/health")
async def health() -> dict:
    return {"ok": True}


authed = APIRouter(dependencies=[Depends(require_key)])


@authed.get("/status")
async def status(request: Request) -> dict:
    s = get_settings()
    store: RecipeStore = request.app.state.recipes
    return {
        "public_url": _public_base(request),
        "openai_base_url": _public_base(request) + "/v1",
        "pod_id": s.runpod_pod_id,
        "gpus": await gpu_stats(),
        "disk": disk_stats(s.models_dir),
        "recipes_repo": s.recipes_repo,
        "recipes_remote_error": store.remote_error,
        "dev": s.mp_dev,
    }


# --- recipes -----------------------------------------------------------------


def _recipe_dict(r, manager: DeploymentManager) -> dict:
    d = manager.deployments.get(r.id)
    return {**r.model_dump(), "deployment_status": d.status if d else None}


@authed.get("/recipes")
async def list_recipes(request: Request) -> list[dict]:
    store: RecipeStore = request.app.state.recipes
    return [_recipe_dict(r, request.app.state.manager) for r in sorted(store.recipes.values(), key=lambda r: r.name)]


@authed.get("/recipes/{recipe_id}")
async def get_recipe(recipe_id: str, request: Request) -> dict:
    store: RecipeStore = request.app.state.recipes
    try:
        return _recipe_dict(store.get(recipe_id), request.app.state.manager)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None


@authed.post("/recipes")
async def push_recipe(request: Request, text: str = Body(..., media_type="text/plain")) -> dict:
    """Add/replace a runtime ("local") recipe from YAML — no rebuild or commit needed."""
    store: RecipeStore = request.app.state.recipes
    try:
        r = store.save_local(text)
    except (ValidationError, ValueError, yaml.YAMLError) as e:
        raise HTTPException(422, str(e)) from None
    return _recipe_dict(r, request.app.state.manager)


@authed.delete("/recipes/{recipe_id}")
async def delete_recipe(recipe_id: str, request: Request) -> dict:
    if not request.app.state.recipes.delete_local(recipe_id):
        raise HTTPException(404, f"no local recipe {recipe_id} (builtin/remote recipes can't be deleted)")
    return {"deleted": recipe_id}


@authed.post("/recipes/refresh")
async def refresh_recipes(request: Request) -> dict:
    s = get_settings()
    store: RecipeStore = request.app.state.recipes
    await store.refresh_remote(s.recipes_repo, s.recipes_ref, s.github_token)
    return {"count": len(store.recipes), "remote_error": store.remote_error}


# --- deployments -------------------------------------------------------------


class DeployRequest(BaseModel):
    recipe_id: str
    params: dict[str, Any] = {}
    force: bool = False


@authed.get("/deployments")
async def list_deployments(request: Request) -> list[dict]:
    base = _public_base(request)
    return [d.to_dict(base) for d in request.app.state.manager.deployments.values()]


@authed.post("/deployments")
async def create_deployment(req: DeployRequest, request: Request) -> dict:
    try:
        recipe = request.app.state.recipes.get(req.recipe_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None
    d = await request.app.state.manager.deploy(recipe, req.params, force=req.force)
    return d.to_dict(_public_base(request))


@authed.get("/deployments/{dep_id}")
async def get_deployment(dep_id: str, request: Request) -> dict:
    return request.app.state.manager.get(dep_id).to_dict(_public_base(request))


@authed.post("/deployments/{dep_id}/stop")
async def stop_deployment(dep_id: str, request: Request, delete_files: bool = True) -> dict:
    d = await request.app.state.manager.stop(dep_id, delete_files=delete_files)
    return d.to_dict(_public_base(request))


@authed.post("/deployments/{dep_id}/restart")
async def restart_deployment(dep_id: str, request: Request, force: bool = False) -> dict:
    """Stop (keeping files, so no re-download) and redeploy with the same params."""
    manager: DeploymentManager = request.app.state.manager
    old = manager.get(dep_id)
    if old.status in ACTIVE:
        await manager.stop(dep_id, delete_files=False)
    try:
        recipe = request.app.state.recipes.get(old.recipe.id)
    except KeyError:
        recipe = old.recipe  # recipe was removed since; reuse the one it ran with
    d = await manager.deploy(recipe, old.params, force=force)
    return d.to_dict(_public_base(request))


@authed.delete("/deployments/{dep_id}")
async def remove_deployment(dep_id: str, request: Request) -> dict:
    request.app.state.manager.remove(dep_id)
    return {"removed": dep_id}


@authed.get("/deployments/{dep_id}/logs")
async def get_logs(dep_id: str, request: Request, since: int = 0) -> list[dict]:
    d = request.app.state.manager.get(dep_id)
    return [{"seq": s, "ts": t, "line": line} for s, t, line in d.logs if s > since]


@authed.get("/deployments/{dep_id}/logs/stream")
async def stream_logs(dep_id: str, request: Request, since: int = 0) -> StreamingResponse:
    d = request.app.state.manager.get(dep_id)

    async def events():
        q = d.subscribe()
        try:
            for s, t, line in list(d.logs):
                if s > since:
                    yield f"data: {json.dumps({'seq': s, 'ts': t, 'line': line})}\n\n"
            while not await request.is_disconnected():
                try:
                    s, t, line = await asyncio.wait_for(q.get(), timeout=15)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if s > since:
                    yield f"data: {json.dumps({'seq': s, 'ts': t, 'line': line})}\n\n"
        finally:
            d.unsubscribe(q)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


api.include_router(authed)
app.include_router(api)


# --- web UI ------------------------------------------------------------------

# Resolved at import (routes must exist before startup) without requiring
# API_KEY, so tests/tools can import the app freely.
_static = Path(os.environ.get("STATIC_DIR") or Settings.model_fields["static_dir"].default)
if _static.is_dir() and (_static / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_static / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        if path.startswith(("api/", "v1/")):
            raise HTTPException(404)
        f = (_static / path).resolve()
        if path and f.is_file() and f.is_relative_to(_static.resolve()):
            return FileResponse(f)
        return FileResponse(_static / "index.html")
