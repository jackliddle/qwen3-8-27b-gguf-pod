from ..recipes import Recipe
from .base import ProcessEngine, RunContext, hf_download


class VllmEngine(ProcessEngine):
    """`vllm serve` on a locally downloaded HF snapshot."""

    name = "vllm"
    known_flags = {
        "ctx_size": "--max-model-len",
        "max_model_len": "--max-model-len",
        "parallel": "--max-num-seqs",
        "max_num_seqs": "--max-num-seqs",
        # Fraction of *total* GPU memory vLLM pre-allocates. Keep this low in
        # recipes if other models should share the GPU.
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "tensor_parallel_size": "--tensor-parallel-size",
        "dtype": "--dtype",
        "quantization": "--quantization",
        "kv_cache_dtype": "--kv-cache-dtype",
        "enforce_eager": "--enforce-eager",
    }

    def validate(self, recipe: Recipe) -> None:
        if not recipe.source.hf_repo:
            raise ValueError("vllm recipes need source.hf_repo")
        self.validate_flags(recipe)

    async def download(self, ctx: RunContext) -> None:
        src = ctx.recipe.source
        await hf_download(ctx, src.hf_repo or "", list(src.files), src.revision)

    def command(self, ctx: RunContext) -> list[str]:
        r = ctx.recipe
        return [
            ctx.settings.vllm_bin, "serve", str(ctx.model_dir),
            "--host", "127.0.0.1",
            "--port", str(ctx.port),
            "--served-model-name", r.served_name,
        ] + self.flag_args(r, ctx.params) + r.extra_args
