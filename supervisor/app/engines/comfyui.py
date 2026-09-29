import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from ..comfy_client import ComfyClient
from ..comfy_workflow import build, check_against_object_info
from ..pngutil import solid_png
from ..recipes import Recipe
from .base import SUPERVISOR_ROOT, Engine, HfFetch, RunContext, Upstream, hf_download_many, terminate

log = logging.getLogger(__name__)

# ComfyUI model folders exposed through extra_model_paths.yaml.
MODEL_DIRS = [
    "checkpoints", "clip", "clip_vision", "configs", "controlnet", "diffusion_models", "unet",
    "embeddings", "loras", "text_encoders", "upscale_models", "vae", "model_patches", "style_models",
]


class ComfyUIEngine(Engine):
    """One shared headless ComfyUI daemon serving every image recipe.

    Each deployment downloads its files into models_dir/<id>/ and links them
    into <comfy_root>/models/<dir>/<id>/, so ComfyUI lists them as
    "<id>/<file>" and recipes never collide on shared filenames (e.g. two
    Qwen recipes both shipping qwen_image_vae.safetensors). ComfyUI loads and
    swaps models per job itself, so several image recipes share one GPU.
    """

    name = "comfyui"
    uses_port = False

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._active: set[str] = set()

    # --- validation -----------------------------------------------------------

    def validate(self, recipe: Recipe) -> None:
        if not isinstance(recipe.workflow, dict) or not recipe.workflow:
            raise ValueError("comfyui recipes need a workflow (API format)")
        if not recipe.source.comfy_files:
            raise ValueError("comfyui recipes need source.comfy_files")
        if "prompt" not in recipe.inputs:
            raise ValueError("comfyui recipes need inputs.prompt")
        if "img2img" in recipe.capabilities and "image" not in recipe.inputs:
            raise ValueError("img2img recipes need inputs.image")
        for field, targets in recipe.inputs.items():
            for t in targets:
                if t.node not in recipe.workflow:
                    raise ValueError(f"inputs.{field} targets node {t.node!r}, not in workflow")
        if recipe.lora_insert:
            for n in (recipe.lora_insert.model_node, recipe.lora_insert.clip_node):
                if n is not None and n not in recipe.workflow:
                    raise ValueError(f"lora_insert node {n!r} not in workflow")
        elif recipe.loras:
            raise ValueError("recipes with loras need lora_insert")
        lp = recipe.params.get("loras")
        if lp is not None:
            unknown = set(lp.values or []) - set(recipe.loras)
            if unknown:
                raise ValueError(f"params.loras values {sorted(unknown)} not in the loras catalog")
        if not any(n.get("class_type") == "SaveImage" for n in recipe.workflow.values()):
            raise ValueError("workflow has no SaveImage node")

    # --- paths / daemon -------------------------------------------------------

    def _base(self, ctx: RunContext) -> str:
        return f"http://127.0.0.1:{ctx.settings.comfyui_port}"

    def _link_dirs(self, ctx: RunContext) -> list[Path]:
        root = ctx.settings.comfy_root / "models"
        return [root / d / ctx.recipe.id for d in MODEL_DIRS]

    def _write_paths_config(self, root: Path) -> Path:
        for d in MODEL_DIRS:
            (root / "models" / d).mkdir(parents=True, exist_ok=True)
        for d in ("input", "output", "temp", "user"):
            (root / d).mkdir(parents=True, exist_ok=True)
        cfg = root / "extra_model_paths.yaml"
        lines = ["model_pod:", f"  base_path: {root / 'models'}", "  is_default: true"]
        lines += [f"  {d}: {d}" for d in MODEL_DIRS]
        cfg.write_text("\n".join(lines) + "\n")
        return cfg

    async def _ensure_daemon(self, ctx: RunContext) -> None:
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                return
            s = ctx.settings
            root = s.comfy_root
            cfg = self._write_paths_config(root)
            if s.comfyui_fake:
                cmd = [sys.executable, "-m", "app.fake_comfy", "--port", str(s.comfyui_port), "--root", str(root)]
                cwd = SUPERVISOR_ROOT
            else:
                cmd = [
                    s.comfyui_python, "main.py",
                    "--listen", "127.0.0.1",
                    "--port", str(s.comfyui_port),
                    "--disable-auto-launch",
                    "--extra-model-paths-config", str(cfg),
                    "--output-directory", str(root / "output"),
                    "--input-directory", str(root / "input"),
                    "--temp-directory", str(root / "temp"),
                    "--user-directory", str(root / "user"),
                ]
                cwd = s.comfyui_dir
            ctx.log("Starting ComfyUI: " + " ".join(cmd))
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                env={**os.environ, **ctx.recipe.env},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._pump = asyncio.create_task(self._log_daemon(self._proc))
            client = ComfyClient(self._base(ctx))
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if self._proc.returncode is not None:
                    raise RuntimeError(f"ComfyUI exited ({self._proc.returncode}); see supervisor logs")
                if await client.alive():
                    ctx.log("ComfyUI is up")
                    return
                await asyncio.sleep(1)
            raise RuntimeError("ComfyUI did not come up within 300s")

    async def _log_daemon(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        async for line in proc.stdout:
            log.info("[comfyui] %s", line.decode(errors="replace").rstrip())

    # --- lifecycle ------------------------------------------------------------

    def _selected_loras(self, ctx: RunContext) -> list[str]:
        return list(ctx.params.get("loras") or [])

    async def download(self, ctx: RunContext) -> None:
        r = ctx.recipe
        # (repo, revision) -> files; each repo goes into its own subdir.
        wanted: dict[tuple[str, str | None], list[str]] = {}
        links: list[tuple[str, str, str, str | None]] = []  # (repo, file, dir, revision)
        for f in r.source.comfy_files:
            wanted.setdefault((f.repo, f.revision), []).append(f.file)
            links.append((f.repo, f.file, f.dir, f.revision))
        for name in self._selected_loras(ctx):
            lo = r.loras[name]
            wanted.setdefault((lo.repo, lo.revision), []).append(lo.file)
            links.append((lo.repo, lo.file, "loras", lo.revision))

        if ctx.settings.comfyui_fake:
            # Dev mode: stand-in files so linking/listing/deleting is exercised.
            for repo, file, _d, _rev in links:
                p = ctx.model_dir / _slug(repo) / file
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\0" * 1024)
            ctx.log(f"Fake download of {len(links)} file(s)")
        else:
            await hf_download_many(
                ctx, [HfFetch(repo, files, rev, subdir=_slug(repo)) for (repo, rev), files in wanted.items()]
            )

        for repo, file, d, _rev in links:
            src = ctx.model_dir / _slug(repo) / file
            if not src.is_file():
                raise RuntimeError(f"expected downloaded file missing: {src}")
            dst = ctx.settings.comfy_root / "models" / d / r.id / Path(file).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        ctx.log(f"Linked {len(links)} file(s) into {ctx.settings.comfy_root / 'models'}")

    async def start(self, ctx: RunContext) -> Upstream:
        await self._ensure_daemon(ctx)
        client = ComfyClient(self._base(ctx))
        r = ctx.recipe

        values = {**ctx.params, "prompt": "a red apple on a wooden table", "negative_prompt": "", "seed": 1, **r.warmup}
        if "img2img" in r.capabilities:
            size = (int(values.get("width") or 512), int(values.get("height") or 512))
            values["image"] = await client.upload(solid_png(*size, (128, 128, 128)), f"mp-warmup-{r.id}.png")
        wf = build(r, values)

        errors = check_against_object_info(wf, await client.object_info())
        if errors:
            raise RuntimeError("workflow doesn't match this ComfyUI:\n  " + "\n  ".join(errors))

        # A real (small) generation loads the weights, so "ready" means the
        # first client request doesn't pay the model load.
        ctx.log(f"Warm-up generation ({', '.join(f'{k}={v}' for k, v in r.warmup.items()) or 'workflow defaults'})")
        t0 = time.monotonic()
        prompt_id = await client.submit(wf)
        await client.wait(prompt_id, timeout=r.start_timeout, check_cancelled=ctx.check_cancelled)
        ctx.log(f"Warm-up done in {time.monotonic() - t0:.1f}s")

        self._active.add(r.id)
        return Upstream(base_url=self._base(ctx), model=r.served_name)

    async def stop(self, ctx: RunContext, upstream: Upstream | None) -> None:
        self._active.discard(ctx.recipe.id)
        if not (self._proc and self._proc.returncode is None):
            return
        if self._active:
            try:
                await ComfyClient(self._base(ctx)).free()
            except Exception as e:  # best effort; files are removed next anyway
                ctx.log(f"ComfyUI /free failed: {e}")
        else:
            ctx.log("No image models left; stopping ComfyUI")
            await self.shutdown()

    async def delete_files(self, ctx: RunContext) -> None:
        for d in self._link_dirs(ctx):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        await super().delete_files(ctx)

    async def shutdown(self) -> None:
        if self._proc:
            await terminate(self._proc)
            if self._pump:
                await self._pump
            self._proc = None


def _slug(repo: str) -> str:
    return repo.replace("/", "--")
