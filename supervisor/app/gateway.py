"""OpenAI-compatible /v1 gateway: routes on the request's `model` to the engine serving it."""

import json

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .auth import require_key

router = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])

# Headers worth passing back from the engine; hop-by-hop and length headers are
# dropped because the body is re-streamed.
_PASS_HEADERS = {"content-type", "cache-control", "x-request-id"}


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
    )


@router.get("/models")
async def list_models(request: Request) -> dict:
    manager = request.app.state.manager
    return {
        "object": "list",
        "data": [
            {
                "id": d.recipe.served_name,
                "object": "model",
                "created": int(d.ready_at or d.created_at),
                "owned_by": d.recipe.engine,
                "recipe_id": d.recipe.id,
                "type": d.recipe.kind,
            }
            for d in manager.deployments.values()
            if d.status == "ready"
        ],
    }


@router.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    manager = request.app.state.manager
    client: httpx.AsyncClient = request.app.state.http
    body = await request.body()

    model: str | None = None
    payload: dict | None = None
    if body and "json" in request.headers.get("content-type", "json"):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return _error(400, "request body is not valid JSON", "invalid_json")
        if isinstance(payload, dict):
            model = payload.get("model")
    else:
        # Multipart (e.g. audio) or GET: allow a header to pick the model.
        model = request.headers.get("x-model")

    dep = manager.route(model)
    if dep is None or dep.upstream is None:
        ready = [d.recipe.served_name for d in manager.deployments.values() if d.status == "ready"]
        if model is None:
            msg = f"specify 'model'; ready models: {ready}"
        else:
            msg = f"model {model!r} is not deployed/ready; ready models: {ready}"
        return _error(404, msg, "model_not_found")

    if dep.recipe.kind == "image":
        return _error(400, f"{dep.recipe.served_name} is an image model; use /v1/images/generations", "wrong_endpoint")

    if payload is not None and payload.get("model") != dep.upstream.model:
        payload["model"] = dep.upstream.model
        body = json.dumps(payload).encode()

    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept")}
    upstream_req = client.build_request(
        request.method,
        f"{dep.upstream.base_url}/v1/{path}",
        content=body,
        headers=headers,
        params={k: v for k, v in request.query_params.items() if k != "key"},
    )
    try:
        resp = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as e:
        return _error(502, f"engine for {dep.recipe.served_name} unreachable: {e}", "upstream_error")

    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items() if k.lower() in _PASS_HEADERS},
        background=BackgroundTask(resp.aclose),
    )
