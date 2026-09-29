"""Deployment lifecycle: pending → downloading → starting → ready → stopping → stopped | failed."""

import asyncio
import logging
import socket
import time
from collections import deque
from typing import Any, Literal

from .config import Settings
from .engines import Cancelled, Engine, Upstream, get_engine
from .recipes import Recipe
from .system import gpu_stats

log = logging.getLogger(__name__)

Status = Literal["pending", "downloading", "starting", "ready", "stopping", "stopped", "failed"]
ACTIVE: set[str] = {"pending", "downloading", "starting", "ready", "stopping"}


class DeployError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class Deployment:
    def __init__(self, recipe: Recipe, params: dict[str, Any], port: int, settings: Settings):
        self.recipe = recipe
        self.params = params
        self.port = port
        self.settings = settings
        self.engine: Engine = get_engine(recipe.engine)
        self.model_dir = settings.models_dir / recipe.id
        self.status: Status = "pending"
        self.error: str | None = None
        self.created_at = time.time()
        self.ready_at: float | None = None
        self.progress: dict[str, int | None] | None = None
        self.upstream: Upstream | None = None
        self.task: asyncio.Task | None = None
        self._watch: asyncio.Task | None = None
        self._cancelled = False
        self._seq = 0
        self.logs: deque[tuple[int, float, str]] = deque(maxlen=3000)
        self._listeners: set[asyncio.Queue] = set()

    @property
    def id(self) -> str:
        return self.recipe.id

    # --- RunContext ---------------------------------------------------------

    def log(self, line: str) -> None:
        self._seq += 1
        entry = (self._seq, time.time(), line.rstrip())
        self.logs.append(entry)
        for q in list(self._listeners):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass

    def set_progress(self, done: int, total: int | None) -> None:
        self.progress = {"done_bytes": done, "total_bytes": total}

    def check_cancelled(self) -> None:
        if self._cancelled:
            raise Cancelled()

    # --- log streaming ------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._listeners.discard(q)

    # --- lifecycle ----------------------------------------------------------

    def _set_status(self, status: Status) -> None:
        self.status = status
        self.log(f"[status] {status}")

    async def run(self) -> None:
        try:
            self._set_status("downloading")
            await self.engine.download(self)
            self.check_cancelled()
            self._set_status("starting")
            self.upstream = await self.engine.start(self)
            self.ready_at = time.time()
            self._set_status("ready")
            self.log(f"Ready in {self.ready_at - self.created_at:.0f}s")
            if self.upstream.process:
                self._watch = asyncio.create_task(self._watch_process(self.upstream.process))
        except (Cancelled, asyncio.CancelledError):
            self.log("Cancelled")
        except Exception as e:
            log.exception("deployment %s failed", self.id)
            self.error = str(e)
            self._set_status("failed")
            self.log(f"ERROR: {e}")

    async def _watch_process(self, proc: asyncio.subprocess.Process) -> None:
        code = await proc.wait()
        if self.status == "ready":
            self.error = f"engine exited unexpectedly (code {code})"
            self._set_status("failed")

    async def stop(self, delete_files: bool = True) -> None:
        self._cancelled = True
        self._set_status("stopping")
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self._watch:
            self._watch.cancel()
        try:
            await self.engine.stop(self, self.upstream)
            if delete_files:
                await self.engine.delete_files(self)
        except Exception as e:
            log.exception("stop %s", self.id)
            self.log(f"ERROR during stop: {e}")
        self.upstream = None
        self._set_status("stopped")

    def to_dict(self, public_base: str) -> dict:
        return {
            "id": self.id,
            "recipe_id": self.recipe.id,
            "name": self.recipe.name,
            "engine": self.recipe.engine,
            "served_name": self.recipe.served_name,
            "capabilities": self.recipe.capabilities,
            "params": self.params,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "ready_at": self.ready_at,
            "progress": self.progress,
            "port": self.port if self.engine.uses_port else None,
            "endpoint": {
                "base_url": f"{public_base}/v1",
                "model": self.recipe.served_name,
            },
        }


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class DeploymentManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.deployments: dict[str, Deployment] = {}

    def active(self) -> list[Deployment]:
        return [d for d in self.deployments.values() if d.status in ACTIVE]

    def _alloc_port(self) -> int:
        used = {d.port for d in self.active()}
        port = self.settings.engine_base_port
        while port in used or not _port_free(port):
            port += 1
        return port

    async def check_vram(self, recipe: Recipe) -> str | None:
        """Return a warning if the recipe's vram_gb likely won't fit, else None."""
        gpus = await gpu_stats()
        if not gpus or not recipe.vram_gb:
            return None
        free_gb = sum(g["memory_free_mib"] for g in gpus) / 1024
        # Deployments still downloading/starting haven't claimed their VRAM yet.
        pending = sum(d.recipe.vram_gb for d in self.active() if d.status in ("pending", "downloading", "starting"))
        avail = free_gb - pending
        if recipe.vram_gb > avail:
            return f"{recipe.id} needs ~{recipe.vram_gb:g}GB VRAM, ~{avail:.1f}GB available"
        return None

    async def deploy(self, recipe: Recipe, overrides: dict[str, Any] | None, force: bool = False) -> Deployment:
        existing = self.deployments.get(recipe.id)
        if existing and existing.status in ACTIVE:
            raise DeployError(f"{recipe.id} is already {existing.status}", 409)
        for d in self.active():
            if d.recipe.served_name == recipe.served_name:
                raise DeployError(f"served_name {recipe.served_name!r} already used by {d.id}", 409)
        try:
            params = recipe.resolve_params(overrides)
        except ValueError as e:
            raise DeployError(str(e)) from e
        warning = await self.check_vram(recipe)
        if warning and not force:
            raise DeployError(warning + " (pass force to deploy anyway)", 409)

        d = Deployment(recipe, params, self._alloc_port(), self.settings)
        if warning:
            d.log(f"WARNING: {warning}")
        self.deployments[recipe.id] = d
        d.task = asyncio.create_task(d.run())
        return d

    def get(self, dep_id: str) -> Deployment:
        try:
            return self.deployments[dep_id]
        except KeyError:
            raise DeployError(f"no deployment {dep_id}", 404) from None

    async def stop(self, dep_id: str, delete_files: bool = True) -> Deployment:
        d = self.get(dep_id)
        if d.status in ("stopped", "stopping"):
            return d
        await d.stop(delete_files=delete_files)
        return d

    def remove(self, dep_id: str) -> None:
        d = self.get(dep_id)
        if d.status in ACTIVE:
            raise DeployError(f"{dep_id} is {d.status}; stop it first", 409)
        del self.deployments[dep_id]

    def route(self, model: str | None) -> Deployment | None:
        ready = [d for d in self.deployments.values() if d.status == "ready"]
        if model is None:
            return ready[0] if len(ready) == 1 else None
        for d in ready:
            if d.recipe.served_name == model:
                return d
        return None

    async def shutdown(self) -> None:
        # Pod is going away; kill engines but don't bother deleting files.
        await asyncio.gather(*(d.stop(delete_files=False) for d in self.active()), return_exceptions=True)
        from .engines.ollama import OllamaEngine

        ollama = get_engine("ollama")
        if isinstance(ollama, OllamaEngine):
            await ollama.shutdown()
