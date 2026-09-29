import asyncio
import sys

from .base import ProcessEngine, RunContext


class FakeEngine(ProcessEngine):
    """Dev/test engine: a tiny OpenAI-ish echo server, no GPU or download needed."""

    name = "fake"

    async def download(self, ctx: RunContext) -> None:
        total = int(ctx.params.get("fake_download_mb", 0) or 0) * 1_000_000
        if not total:
            return
        ctx.model_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 11):
            ctx.check_cancelled()
            await asyncio.sleep(0.3)
            ctx.set_progress(total * i // 10, total)
        (ctx.model_dir / "weights.bin").write_bytes(b"\0" * 1024)
        ctx.log("Fake download complete")

    def command(self, ctx: RunContext) -> list[str]:
        cmd = [sys.executable, "-m", "app.fake_server", "--port", str(ctx.port), "--model", ctx.recipe.served_name]
        if ctx.params.get("fail_start"):
            cmd.append("--fail")
        return cmd
