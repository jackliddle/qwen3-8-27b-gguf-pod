# model-pod — notes for Claude

## Adding a model (agentic workflow)
1. Inspect the HF repo (file list + sizes) and pick the files/quant; estimate `vram_gb`
   (weights + KV cache for the default ctx) and `disk_gb`.
2. Write `recipes/<id>.yaml` (id == filename). Set `capabilities` honestly — they pick the smoke tests.
3. `cd supervisor && API_KEY=test poetry run pytest` (validates every recipe; engine command snapshots).
4. Deploy on a live pod: `scripts/pod.sh status` (or `up`; state the $/hr first), `eval "$(scripts/pod.sh env)"`,
   `poetry run mp push ../recipes/<id>.yaml` (no commit/rebuild needed), `poetry run mp deploy <id> --wait`.
5. `poetry run mp test <id>`; on failure `mp logs <id>` and adjust params/extra_args, `mp push`, `mp restart <id>`.
6. `mp stop <id>`; commit the recipe; `scripts/pod.sh down` if the pod isn't needed (it bills while up).

## Adding an image model
Same loop, but the recipe needs a ComfyUI API-format workflow (`recipes/workflows/<id>.json`):
- Start from ComfyUI's official template: `pip download comfyui-workflow-templates-json`, find
  `templates/image_<model>*.json`. These are UI-format, often with a subgraph, so hand-convert the
  subgraph's nodes to API format. Take input names from the node's `define_schema` in the pinned
  ComfyUI source (see Dockerfile `COMFYUI_TAG`). The template's `properties.models` lists the exact HF URLs.
- Map request fields in `inputs:`. Set `lora_insert.model_node` to the UNET loader. Mark `txt2img` and/or `img2img`.
- `pytest` validates node references offline. On deploy, `/object_info` validation plus a warm-up generation
  catch the rest. `mp test <id> --save /tmp/imgs`, then look at the PNGs before calling it done.

## Policy (carried over from runpod-endpoints/ENDPOINTS.md)
- SECURE cloud only, never COMMUNITY. Recreate rather than resume; retry on "no instances available".
- Only terminate pods this session created or that `.pod` records.

## Layout
- `supervisor/app/` — FastAPI supervisor: `recipes.py` (schema/loader), `deployments.py` (lifecycle),
  `gateway.py` (/v1 proxy), `engines/` (one module per engine), `smoke.py`, `cli.py` (`mp`).
- `frontend/` — React/Vite UI, built into the image and served by the supervisor.
- `scripts/pod.sh` — RunPod REST v1 create/status/terminate.
