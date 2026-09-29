"""Fake ComfyUI for dev/tests (COMFYUI_FAKE=1): the subset of the HTTP API the
supervisor uses, producing solid-colour PNGs of the requested size."""

import argparse
import asyncio
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from .pngutil import png_size, solid_png

# Node classes this fake "has". Anything else is rejected like a missing custom node.
KNOWN = {
    "SaveImage", "CLIPLoader", "VAELoader", "UNETLoader", "CLIPTextEncode", "VAEDecode", "VAEEncode",
    "EmptyLatentImage", "EmptySD3LatentImage", "KSampler", "ModelSamplingAuraFlow", "CFGNorm",
    "PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean", "ComfySwitchNode", "ConditioningZeroOut",
    "LoraLoader", "LoraLoaderModelOnly", "LoadImage", "TextEncodeQwenImageEditPlus",
    "FluxKontextMultiReferenceLatentMethod", "FluxKontextImageScale", "ImageScaleToTotalPixels",
}
# name input → models/<folder> it lists (mirrors ComfyUI's loaders)
LISTS = {"unet_name": "diffusion_models", "clip_name": "text_encoders", "vae_name": "vae", "lora_name": "loras"}


def build_app(root: Path) -> FastAPI:
    app = FastAPI()
    history: dict[str, dict] = {}

    def listing(folder: str) -> list[str]:
        base = root / "models" / folder
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()) if base.exists() else []

    @app.get("/system_stats")
    async def system_stats() -> dict:
        return {"system": {"comfyui_version": "fake"}, "devices": []}

    @app.get("/object_info")
    async def object_info() -> dict:
        info = {}
        for cls in KNOWN:
            required = {name: [listing(folder)] for name, folder in LISTS.items()}
            if cls == "LoadImage":
                required = {"image": [sorted(p.name for p in (root / "input").glob("*"))]}
            info[cls] = {"input": {"required": required}}
        return info

    @app.post("/prompt")
    async def prompt(request: Request):
        wf = (await request.json())["prompt"]
        node_errors = {}
        for nid, node in wf.items():
            cls = node.get("class_type")
            if cls not in KNOWN:
                node_errors[nid] = {"class_type": cls, "errors": [{"message": f"unknown node {cls}"}]}
            for key, folder in LISTS.items():
                val = node.get("inputs", {}).get(key)
                if isinstance(val, str) and val not in listing(folder):
                    node_errors[nid] = {"class_type": cls, "errors": [{"message": "value not in list", "details": f"{key}: {val}"}]}
        if node_errors:
            return JSONResponse(status_code=400, content={"error": {"message": "Prompt outputs failed validation"}, "node_errors": node_errors})
        pid = uuid.uuid4().hex
        asyncio.get_running_loop().call_later(0.2, lambda: _finish(pid, wf))
        return {"prompt_id": pid, "number": len(history)}

    def _finish(pid: str, wf: dict) -> None:
        width = height = 512
        batch = 1
        color = (255, 0, 0)
        for node in wf.values():
            inp = node.get("inputs", {})
            if node["class_type"] in ("EmptyLatentImage", "EmptySD3LatentImage"):
                width, height, batch = inp.get("width", 512), inp.get("height", 512), inp.get("batch_size", 1)
            if node["class_type"] == "LoadImage":
                # "Edit": keep the input's size, change the colour.
                width, height = png_size((root / "input" / inp["image"]).read_bytes())
                color = (0, 0, 255)
        outputs = {}
        for nid, node in wf.items():
            if node["class_type"] != "SaveImage":
                continue
            prefix = node["inputs"].get("filename_prefix", "ComfyUI")
            sub, _, stem = prefix.rpartition("/")
            (root / "output" / sub).mkdir(parents=True, exist_ok=True)
            images = []
            for i in range(batch):
                name = f"{stem}_{i + 1:05d}_.png"
                (root / "output" / sub / name).write_bytes(solid_png(width, height, color))
                images.append({"filename": name, "subfolder": sub, "type": "output"})
            outputs[nid] = {"images": images}
        history[pid] = {"outputs": outputs, "status": {"status_str": "success", "completed": True, "messages": []}}

    @app.get("/history/{pid}")
    async def get_history(pid: str) -> dict:
        return {pid: history[pid]} if pid in history else {}

    @app.get("/view")
    async def view(filename: str, subfolder: str = "", type: str = "output") -> Response:
        return Response((root / type / subfolder / filename).read_bytes(), media_type="image/png")

    @app.post("/upload/image")
    async def upload(image: UploadFile = File(...), overwrite: str = Form("false"), type: str = Form("input")) -> dict:
        (root / "input").mkdir(parents=True, exist_ok=True)
        (root / "input" / image.filename).write_bytes(await image.read())
        return {"name": image.filename, "subfolder": "", "type": "input"}

    @app.post("/free")
    async def free() -> dict:
        return {}

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--root", required=True)
    a = p.parse_args()
    uvicorn.run(build_app(Path(a.root)), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
