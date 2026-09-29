from pathlib import Path

from ..recipes import Recipe
from .base import ProcessEngine, RunContext, hf_download


class LlamaCppEngine(ProcessEngine):
    """llama.cpp's llama-server, one process per GGUF model."""

    name = "llamacpp"
    known_flags = {
        "ctx_size": "--ctx-size",
        "parallel": "--parallel",
        "n_gpu_layers": "--n-gpu-layers",
        "threads": "--threads",
        "batch_size": "--batch-size",
        "ubatch_size": "--ubatch-size",
        "cache_type_k": "--cache-type-k",
        "cache_type_v": "--cache-type-v",
        "flash_attn": "--flash-attn",
    }

    def validate(self, recipe: Recipe) -> None:
        if not recipe.source.hf_repo or not recipe.source.files:
            raise ValueError("llamacpp recipes need source.hf_repo and source.files")
        self.validate_flags(recipe)

    async def download(self, ctx: RunContext) -> None:
        src = ctx.recipe.source
        patterns = list(src.files) + ([src.mmproj] if src.mmproj else [])
        await hf_download(ctx, src.hf_repo or "", patterns, src.revision)

    def _resolve(self, model_dir: Path, pattern: str, exclude: str | None = None) -> Path:
        excluded = set(model_dir.glob(exclude)) if exclude else set()
        matches = sorted(p for p in model_dir.glob(pattern) if p.is_file() and p not in excluded)
        if not matches:
            raise RuntimeError(f"no downloaded file matches {pattern!r} in {model_dir}")
        return matches[0]

    def command(self, ctx: RunContext) -> list[str]:
        r = ctx.recipe
        model = self._resolve(ctx.model_dir, r.source.files[0], exclude=r.source.mmproj)
        cmd = [
            ctx.settings.llama_server_bin,
            "--model", str(model),
            "--host", "127.0.0.1",
            "--port", str(ctx.port),
            "--alias", r.served_name,
        ]
        if r.source.mmproj:
            cmd += ["--mmproj", str(self._resolve(ctx.model_dir, r.source.mmproj))]
        if "n_gpu_layers" not in r.params:
            cmd += ["--n-gpu-layers", "999"]
        return cmd + self.flag_args(r, ctx.params) + r.extra_args
