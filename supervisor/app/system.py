import asyncio
import shutil
from pathlib import Path


async def gpu_stats() -> list[dict]:
    """Per-GPU name and memory in MiB from nvidia-smi; [] when there is no GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    gpus = []
    for line in out.decode().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        idx, name, total, used, free, util = parts
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_pct": int(util) if util.isdigit() else None,
            }
        )
    return gpus


def disk_stats(path: Path) -> dict:
    # models_dir may not exist yet; measure the filesystem it will live on.
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    u = shutil.disk_usage(p)
    return {"path": str(path), "total_bytes": u.total, "used_bytes": u.used, "free_bytes": u.free}
