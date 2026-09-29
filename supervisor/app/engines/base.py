import asyncio
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..config import Settings
from ..recipes import Recipe

# `python -m app.<module>` child processes need to run from here.
SUPERVISOR_ROOT = Path(__file__).resolve().parents[2]


class Cancelled(Exception):
    pass


class RunContext(Protocol):
    """What an engine sees of a deployment (implemented by deployments.Deployment)."""

    recipe: Recipe
    params: dict[str, Any]
    model_dir: Path
    port: int
    settings: Settings

    def log(self, line: str) -> None: ...
    def set_progress(self, done: int, total: int | None) -> None: ...
    def check_cancelled(self) -> None: ...


@dataclass
class Upstream:
    base_url: str  # e.g. http://127.0.0.1:9001 (gateway appends /v1/...)
    model: str  # model name the engine expects in request bodies
    process: asyncio.subprocess.Process | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Engine(ABC):
    name: str
    # Flag used for well-known param names when a recipe doesn't give one.
    known_flags: dict[str, str] = {}
    uses_port = True

    def validate(self, recipe: Recipe) -> None:
        """Raise ValueError if the recipe can't work with this engine."""

    @abstractmethod
    async def download(self, ctx: RunContext) -> None: ...

    @abstractmethod
    async def start(self, ctx: RunContext) -> Upstream: ...

    @abstractmethod
    async def stop(self, ctx: RunContext, upstream: Upstream | None) -> None: ...

    async def delete_files(self, ctx: RunContext) -> None:
        if ctx.model_dir.exists():
            await asyncio.to_thread(shutil.rmtree, ctx.model_dir, ignore_errors=True)
            ctx.log(f"Deleted {ctx.model_dir}")

    def flag_args(self, recipe: Recipe, params: dict[str, Any]) -> list[str]:
        args: list[str] = []
        for name, spec in recipe.params.items():
            flag = spec.flag or self.known_flags.get(name)
            value = params.get(name)
            if flag is None or value is None:
                continue
            if spec.type == "bool":
                if value:
                    args.append(flag)
            else:
                args += [flag, str(value)]
        return args

    def validate_flags(self, recipe: Recipe) -> None:
        missing = [n for n, s in recipe.params.items() if not (s.flag or n in self.known_flags)]
        if missing:
            raise ValueError(f"params {missing} have no 'flag' and aren't known to engine {self.name}")


# --- helpers shared by process-per-model engines -----------------------------


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _expected_hf_size(repo: str, revision: str | None, patterns: list[str]) -> tuple[int, int]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    total = count = 0
    for item in HfApi().list_repo_tree(repo, revision=revision, recursive=True):
        if not isinstance(item, RepoFile):
            continue
        if patterns and not any(fnmatch(item.path, p) for p in patterns):
            continue
        total += item.size or 0
        count += 1
    return total, count


async def hf_download(ctx: RunContext, repo: str, patterns: list[str], revision: str | None) -> None:
    """Fetch (part of) an HF repo into ctx.model_dir, with retries and progress.

    Runs snapshot_download in a child process so a Stop mid-download can kill
    it. snapshot_download stages into <dir>/.cache and only materialises each
    file once complete, so interrupted attempts never leave a corrupt file and
    a retry resumes.
    """
    ctx.model_dir.mkdir(parents=True, exist_ok=True)
    total, count = await asyncio.to_thread(_expected_hf_size, repo, revision, patterns)
    if count == 0:
        raise RuntimeError(f"no files in {repo} match {patterns or '*'}")
    free = shutil.disk_usage(ctx.model_dir).free
    if total > free:
        raise RuntimeError(f"need {total / 1e9:.1f}GB, only {free / 1e9:.1f}GB free on {ctx.model_dir}")
    ctx.log(f"Downloading {count} file(s), {total / 1e9:.2f}GB from {repo}")
    ctx.set_progress(dir_size(ctx.model_dir), total)

    cmd = [sys.executable, "-m", "app.hf_fetch", "--repo", repo, "--dir", str(ctx.model_dir)]
    if revision:
        cmd += ["--revision", revision]
    for p in patterns:
        cmd += ["--pattern", p]

    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        ctx.log(f"Download attempt {attempt}/{max_attempts}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=SUPERVISOR_ROOT,
        )
        pump = asyncio.create_task(_pump(proc, ctx))
        try:
            while proc.returncode is None:
                ctx.set_progress(dir_size(ctx.model_dir), total)
                ctx.check_cancelled()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except TimeoutError:
                    pass
        except BaseException:
            await terminate(proc)
            raise
        finally:
            await pump
        if proc.returncode == 0:
            ctx.set_progress(total, total)
            ctx.log("Download complete")
            return
        ctx.log(f"Attempt {attempt} failed (exit {proc.returncode})")
        await asyncio.sleep(5)
    raise RuntimeError(f"download failed after {max_attempts} attempts")


async def _pump(proc: asyncio.subprocess.Process, ctx: RunContext) -> None:
    assert proc.stdout is not None
    # tqdm redraws with \r; split on both so the log gets readable lines, and
    # keep at most one progress-bar line per 15s (the UI has a real progress bar).
    buf = b""
    last_bar = 0.0
    while chunk := await proc.stdout.read(4096):
        buf += chunk
        *lines, buf = buf.replace(b"\r", b"\n").split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            text = line.decode(errors="replace")
            if "%|" in text and "100%|" not in text:
                if time.monotonic() - last_bar < 15:
                    continue
                last_bar = time.monotonic()
            ctx.log(text)
    if buf.strip():
        ctx.log(buf.decode(errors="replace"))


async def terminate(proc: asyncio.subprocess.Process, grace: float = 30) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except TimeoutError:
        proc.kill()
        await proc.wait()


class ProcessEngine(Engine):
    """One OS process per deployment, speaking OpenAI on 127.0.0.1:<port>."""

    health_path = "/health"

    @abstractmethod
    def command(self, ctx: RunContext) -> list[str]: ...

    def process_env(self, ctx: RunContext) -> dict[str, str]:
        return {**os.environ, **ctx.recipe.env}

    async def start(self, ctx: RunContext) -> Upstream:
        cmd = self.command(ctx)
        ctx.log("$ " + " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self.process_env(ctx),
            cwd=SUPERVISOR_ROOT,
            start_new_session=True,
        )
        upstream = Upstream(base_url=f"http://127.0.0.1:{ctx.port}", model=ctx.recipe.served_name, process=proc)
        upstream.extra["pump"] = asyncio.create_task(_pump(proc, ctx))
        try:
            await self._wait_healthy(ctx, upstream)
        except BaseException:
            await terminate(proc)
            raise
        return upstream

    async def _wait_healthy(self, ctx: RunContext, up: Upstream) -> None:
        assert up.process is not None
        deadline = time.monotonic() + ctx.recipe.start_timeout
        async with httpx.AsyncClient(timeout=5) as c:
            while time.monotonic() < deadline:
                ctx.check_cancelled()
                if up.process.returncode is not None:
                    raise RuntimeError(f"engine exited with code {up.process.returncode} during startup")
                try:
                    r = await c.get(up.base_url + self.health_path)
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2)
        raise RuntimeError(f"engine not healthy after {ctx.recipe.start_timeout}s")

    async def stop(self, ctx: RunContext, upstream: Upstream | None) -> None:
        if upstream and upstream.process:
            await terminate(upstream.process)
            pump = upstream.extra.get("pump")
            if pump:
                await pump
