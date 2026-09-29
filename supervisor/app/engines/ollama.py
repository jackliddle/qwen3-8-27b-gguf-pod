import asyncio
import json
import logging
import os

import httpx

from ..recipes import Recipe
from .base import Engine, RunContext, Upstream, terminate

log = logging.getLogger(__name__)


class OllamaEngine(Engine):
    """One shared `ollama serve` daemon; each deployment is a derived model.

    Recipe params become Modelfile PARAMETERs (num_ctx, temperature, ...) via
    `ollama create <served_name> --from <tag>`, so clients use served_name and
    get the configured context size without any request rewriting.
    """

    name = "ollama"
    uses_port = False

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def validate(self, recipe: Recipe) -> None:
        if not recipe.source.ollama_model:
            raise ValueError("ollama recipes need source.ollama_model")

    def _base(self, ctx: RunContext) -> str:
        return f"http://127.0.0.1:{ctx.settings.ollama_port}"

    async def _ensure_daemon(self, ctx: RunContext) -> None:
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                return
            s = ctx.settings
            env = {
                **os.environ,
                "OLLAMA_HOST": f"127.0.0.1:{s.ollama_port}",
                "OLLAMA_MODELS": str(s.models_dir / "_ollama"),
                "OLLAMA_KEEP_ALIVE": "-1",
            }
            ctx.log("Starting ollama daemon")
            self._proc = await asyncio.create_subprocess_exec(
                s.ollama_bin, "serve",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._pump = asyncio.create_task(self._log_daemon(self._proc))
            async with httpx.AsyncClient(timeout=2) as c:
                for _ in range(60):
                    if self._proc.returncode is not None:
                        raise RuntimeError(f"ollama daemon exited ({self._proc.returncode})")
                    try:
                        if (await c.get(self._base(ctx))).status_code == 200:
                            return
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
            raise RuntimeError("ollama daemon did not come up")

    async def _log_daemon(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        async for line in proc.stdout:
            log.info("[ollama] %s", line.decode(errors="replace").rstrip())

    async def download(self, ctx: RunContext) -> None:
        await self._ensure_daemon(ctx)
        tag = ctx.recipe.source.ollama_model
        ctx.log(f"ollama pull {tag}")
        layers: dict[str, tuple[int, int]] = {}
        last_status = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=600)) as c:
            async with c.stream("POST", self._base(ctx) + "/api/pull", json={"model": tag, "stream": True}) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    ctx.check_cancelled()
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    if "error" in ev:
                        raise RuntimeError(f"ollama pull: {ev['error']}")
                    if ev.get("digest") and ev.get("total"):
                        layers[ev["digest"]] = (ev.get("completed", 0), ev["total"])
                        ctx.set_progress(sum(d for d, _ in layers.values()), sum(t for _, t in layers.values()))
                    if ev.get("status") != last_status:
                        last_status = ev.get("status")
                        ctx.log(f"pull: {last_status}")

    async def start(self, ctx: RunContext) -> Upstream:
        await self._ensure_daemon(ctx)
        r = ctx.recipe
        parameters = {k: v for k, v in ctx.params.items() if v is not None}
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=ctx.recipe.start_timeout)) as c:
            ctx.log(f"ollama create {r.served_name} from {r.source.ollama_model} {parameters}")
            resp = await c.post(
                self._base(ctx) + "/api/create",
                json={"model": r.served_name, "from": r.source.ollama_model, "parameters": parameters, "stream": False},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"ollama create failed: {resp.text}")
            ctx.log("Loading model into VRAM")
            resp = await c.post(self._base(ctx) + "/api/generate", json={"model": r.served_name, "keep_alive": -1})
            if resp.status_code != 200:
                raise RuntimeError(f"ollama load failed: {resp.text}")
        return Upstream(base_url=self._base(ctx), model=r.served_name)

    async def stop(self, ctx: RunContext, upstream: Upstream | None) -> None:
        if not (self._proc and self._proc.returncode is None):
            return
        async with httpx.AsyncClient(timeout=60) as c:
            try:
                await c.post(self._base(ctx) + "/api/generate", json={"model": ctx.recipe.served_name, "keep_alive": 0})
            except httpx.HTTPError as e:
                ctx.log(f"unload failed: {e}")

    async def delete_files(self, ctx: RunContext) -> None:
        if not (self._proc and self._proc.returncode is None):
            return
        async with httpx.AsyncClient(timeout=60) as c:
            # Removing the derived model then the base tag frees the blobs;
            # ollama refcounts layers, so other deployments of the same tag
            # keep theirs.
            for name in (ctx.recipe.served_name, ctx.recipe.source.ollama_model):
                try:
                    resp = await c.request("DELETE", self._base(ctx) + "/api/delete", json={"model": name})
                    ctx.log(f"ollama rm {name}: {resp.status_code}")
                except httpx.HTTPError as e:
                    ctx.log(f"ollama rm {name} failed: {e}")

    async def shutdown(self) -> None:
        if self._proc:
            await terminate(self._proc)
