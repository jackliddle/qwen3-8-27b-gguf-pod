"""Async client for the bits of ComfyUI's HTTP API the supervisor uses."""

import asyncio
import time
from typing import Callable

import httpx


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def alive(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=3) as c:
                return (await c.get("/system_stats")).status_code == 200
        except httpx.HTTPError:
            return False

    async def object_info(self) -> dict:
        async with self._client() as c:
            r = await c.get("/object_info")
            r.raise_for_status()
            return r.json()

    async def submit(self, workflow: dict) -> str:
        async with self._client() as c:
            r = await c.post("/prompt", json={"prompt": workflow, "client_id": "model-pod"})
        if r.status_code != 200:
            # ComfyUI explains validation failures in error + node_errors.
            try:
                body = r.json()
                detail = body.get("error", {}).get("message") or str(body)
                for nid, ne in (body.get("node_errors") or {}).items():
                    msgs = "; ".join(e.get("details") or e.get("message", "") for e in ne.get("errors", []))
                    detail += f" | node {nid} ({ne.get('class_type')}): {msgs}"
            except ValueError:
                detail = r.text
            raise ComfyError(f"ComfyUI rejected the workflow: {detail}")
        return r.json()["prompt_id"]

    async def history(self, prompt_id: str) -> dict | None:
        async with self._client() as c:
            r = await c.get(f"/history/{prompt_id}")
            r.raise_for_status()
            return r.json().get(prompt_id)

    async def wait(
        self,
        prompt_id: str,
        timeout: float,
        check_cancelled: Callable[[], None] | None = None,
        interval: float = 0.5,
    ) -> dict:
        """Poll until the prompt finishes; returns its history entry or raises ComfyError."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if check_cancelled:
                check_cancelled()
            entry = await self.history(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyError(_execution_error(status))
                if status.get("completed", True):
                    return entry
            await asyncio.sleep(interval)
        raise ComfyError(f"prompt {prompt_id} not finished after {timeout:.0f}s")

    async def view(self, image: dict) -> bytes:
        async with self._client() as c:
            r = await c.get(
                "/view",
                params={"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")},
            )
            r.raise_for_status()
            return r.content

    async def upload(self, data: bytes, filename: str) -> str:
        """Upload an input image; returns the name to put in a LoadImage node."""
        async with self._client() as c:
            r = await c.post(
                "/upload/image",
                files={"image": (filename, data, "image/png")},
                data={"overwrite": "true", "type": "input"},
            )
            r.raise_for_status()
            body = r.json()
        return f"{body['subfolder']}/{body['name']}" if body.get("subfolder") else body["name"]

    async def free(self) -> None:
        async with self._client() as c:
            await c.post("/free", json={"unload_models": True, "free_memory": True})


def output_images(entry: dict) -> list[dict]:
    """Saved images (not previews/temp) from a history entry, in node order."""
    out = []
    for node_out in (entry.get("outputs") or {}).values():
        for img in node_out.get("images", []):
            if img.get("type", "output") == "output":
                out.append(img)
    return out


def _execution_error(status: dict) -> str:
    for event, data in status.get("messages") or []:
        if event == "execution_error":
            return f"{data.get('node_type')} (node {data.get('node_id')}): {data.get('exception_message', '').strip()}"
    return "ComfyUI reported an execution error"
