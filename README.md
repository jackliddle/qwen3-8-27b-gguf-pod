# model-pod

One RunPod GPU pod image that runs a **supervisor + web UI** for deploying
LLMs on demand. Pick a model recipe, adjust its options, deploy, and copy the
endpoint. Models are downloaded when deployed and deleted when stopped.
Terminate the pod when you're done and nothing is billed.

```
RunPod GPU pod (ghcr.io/jackliddle/model-pod)
└─ supervisor (FastAPI, :8000)   ← the only public port
   ├─ /       React UI
   ├─ /api/*  recipes, deployments, logs, GPU/disk stats
   ├─ /v1/*   OpenAI-compatible gateway, routed by the request's "model"
   └─ engines on 127.0.0.1: llama-server | vllm serve | ollama serve
```

Engines: **llama.cpp** (GGUF), **vLLM** (HF safetensors/AWQ/FP8) and
**Ollama** for LLMs, plus headless **ComfyUI** for image models (Qwen-Image,
Krea 2, Qwen-Image-Edit; see [Image models](#image-models)). Several models can run at once if VRAM allows. The supervisor
checks each recipe's `vram_gb` against free VRAM before deploying.

## Quick start

```bash
export RUNPOD_API_KEY=...            # HF_TOKEN / PUBLIC_KEY optional (gated models / SSH)
scripts/pod.sh up                    # SECURE cloud, A40 → A6000 → 6000 Ada → L40S
# prints the UI login link (…/#key=…), the OpenAI base URL and the API key
eval "$(scripts/pod.sh env)"         # MP_URL + MP_API_KEY for the CLI
scripts/pod.sh down                  # terminate: stops all billing
```

Every client uses one base URL, `https://<pod-id>-8000.proxy.runpod.net/v1`,
with the pod's `API_KEY` as the bearer token, and picks a model by its
`served_name`.

## Recipes (`recipes/*.yaml`)

A recipe is the plugin unit: which engine, what to download, and which
options the UI shows (with their defaults).

```yaml
id: qwen3.8-27b-hauhau-q8            # must match the filename
name: Qwen3.8 27B Uncensored Aggressive (Q8_K_P, MTP)
engine: llamacpp                     # llamacpp | vllm | ollama
source:
  hf_repo: HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
  files: [Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf]   # globs OK
  mmproj: mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf
served_name: qwen3.8-27b             # the "model" clients send
vram_gb: 38
disk_gb: 33
capabilities: [chat, vision, reasoning]   # also selects the smoke tests
params:                              # UI form; unknown names need a `flag`
  ctx_size: {type: int, default: 32768}
  reasoning_effort: {type: enum, values: [low, medium, high], default: medium, flag: --reasoning-effort}
extra_args: [--jinja, --spec-type, draft-mtp]
```

- **llamacpp**: `source.files` lists the files/globs to fetch. The first match
  is `--model` (for split GGUFs, list the `-00001-of-` shard). `mmproj` is
  optional.
- **vllm**: `source.files` are allow-patterns for the snapshot (empty means
  the whole repo). Keep `gpu_memory_utilization` low if the GPU will be
  shared.
- **ollama**: `source.ollama_model` is a library tag or `hf.co/<repo>:<quant>`.
  Params become Modelfile `PARAMETER`s (e.g. `num_ctx`).
- Common params map to engine flags automatically (`ctx_size` becomes
  `--ctx-size` for llama.cpp or `--max-model-len` for vLLM). Anything else
  takes a `flag:` or goes in `extra_args`.

Recipes are loaded in layers, later ones winning: those baked into the image,
then the repo's `main` branch (re-fetched at startup and by the UI's
**Refresh** button, via `RECIPES_REPO`), then recipes pushed at runtime
(**Add recipe** / `mp push`). A new recipe never needs an image rebuild.

## Image models

Image recipes use `engine: comfyui`. The supervisor runs one headless ComfyUI
(no UI exposed, started on the first image deploy and stopped with the last),
which loads and swaps models per job, so several image models can be deployed
at once. Each recipe downloads Comfy-Org's fp8 repackages and links them into
ComfyUI as `<recipe-id>/<file>`, so recipes never clash over shared filenames.

| Recipe | Model | Notes |
|---|---|---|
| `krea-2-turbo` | Krea 2 Turbo fp8, 8 steps | 9 optional style LoRAs. Krea community licence. |
| `krea-2-raw` | Krea 2 Raw fp8, 52 steps, cfg 3.5 | Undistilled base. Same LoRAs. |
| `qwen-image-2512` | Qwen-Image-2512 fp8 | `fast` = 4-step Lightning LoRA. |
| `qwen-image-edit-2511` | Qwen-Image-Edit-2511 fp8 | Instruction editing of one input image; `fast` as above. |

**API**, on the same base URL and key as the LLMs:

```bash
# OpenAI-compatible (synchronous: fine for Turbo/Lightning renders)
curl $BASE/v1/images/generations -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "krea-2-turbo", "prompt": "a red fox in the snow", "size": "1024x1024",
       "seed": 42, "loras": [{"name": "neondrip", "strength": 0.8}]}'
curl $BASE/v1/images/edits -H "Authorization: Bearer $KEY" \
  -F model=qwen-image-edit-2511 -F image=@in.png -F prompt="make it night time"

# Async jobs: anything that might exceed RunPod's ~100s proxy timeout
curl $BASE/api/images/jobs -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen-image-2512", "prompt": "...", "params": {"fast": false}}'   # -> {"job_id": ...}
curl $BASE/api/images/jobs/<job_id> -H "Authorization: Bearer $KEY"             # -> status, images[].url
```

- **Extra request fields:** `negative_prompt`, `seed`, `steps`, `cfg`,
  `loras`, `params` (recipe param overrides such as `fast`), and `image_b64`
  (for jobs on edit models). Responses include the `seed` used.
- **LoRAs:** pick them in the deploy dialog (or `mp deploy krea-2-turbo --set
  loras=neondrip,darkbrush`); only selected ones are downloaded. A style
  LoRA's trigger phrase is appended to the prompt automatically.
- **Result URLs** (`/api/images/files/...`) need no key: each name contains a
  random 128-bit job id. They last until the pod is terminated.

**Adding an image model** means a recipe plus a ComfyUI API-format workflow
in `recipes/workflows/`. In ComfyUI, *Export (API)* a working workflow, or
adapt one of ComfyUI's built-in templates. The recipe then lists:
- `comfy_files`: repo, file, and the ComfyUI model folder for each file.
- `inputs`: which node/input each request field sets. List several targets
  when a value exists once per branch, e.g. steps behind the Lightning switch.
- `lora_insert.model_node`: where LoRA chains are spliced in.

On deploy the workflow is checked against ComfyUI's `/object_info`, which
catches unknown node types and wrong filenames before "ready". A warm-up
generation then loads the weights.

## CLI: `mp`

`mp` is installed on the pod, or run it locally from `supervisor/` with
`poetry run mp`. It reads `MP_URL` and `MP_API_KEY`, and every command exits
non-zero on failure.

```
mp status                         GPU / disk / URLs
mp recipes                        list recipes
mp push my-model.yaml             add or replace a runtime recipe
mp deploy <id> [--set k=v] --wait deploy and block until ready (prints connection info)
mp test <id> [--save DIR]         smoke suite: LLMs: models, chat, stream + vision/tools/reasoning;
                                  images: generate, async job, a LoRA, edits (images saved to DIR)
mp gen <model> "prompt" -o x.png  one image via the job API [--size --steps --seed --lora n:0.8 --image in.png --set fast=true]
mp logs <id> [-f]                 engine / download logs
mp endpoints [<id>]               copy-pasteable env vars + curl
mp stop <id> [--keep-files]       stop; deletes the model files by default
mp restart <id>                   same params, keeps files
```

## Development (no GPU needed)

```bash
cd supervisor && poetry install
MP_DEV=1 COMFYUI_FAKE=1 API_KEY=dev MODELS_DIR=/tmp/mp/models LOCAL_RECIPES_DIR=/tmp/mp/recipes \
  COMFY_ROOT=/tmp/mp/comfy poetry run uvicorn app.main:app --port 8000
  # dev mode adds fake engines: an echo LLM, and a fake ComfyUI returning solid PNGs
cd frontend && npm install && npm run dev          # proxies /api and /v1 to :8000
API_KEY=test poetry run pytest                     # unit + full-stack tests against the fake engine
```

CI (`.github/workflows/build.yml`) runs the tests, lint and frontend build,
then builds and pushes `ghcr.io/<owner>/model-pod:{latest,sha}`.

## Image notes

- The base is `vllm/vllm-openai:v0.30.0-ubuntu2404` (CUDA 13.0). llama.cpp's
  binaries come from `ghcr.io/ggml-org/llama.cpp:server-cuda13-b10524`, which
  has upstream MTP support, together with that image's own CUDA 13.3 cuBLAS
  and cudart in `/opt/llama.cpp/cuda`. They are only on llama-server's
  library path. Ollama comes from `ollama/ollama:0.35.0` with its bundled
  libs. ComfyUI `v0.37.0` is cloned to `/opt/ComfyUI` with a venv that reuses
  vLLM's system torch; only ComfyUI's other dependencies go into the venv. Pods need a CUDA 13 host driver, and `pod.sh` filters for one.
- The supervisor runs in its own venv so its dependencies never conflict with
  vLLM's.
- Downloads use `huggingface_hub` (hf_xet) in a killable child process with
  10 retries. Progress in the UI comes from the size of the download
  directory against the repo's file sizes.
- Set `HF_TOKEN` on the pod for gated repos and higher rate limits.
- RunPod's HTTP proxy times out non-streaming requests at about 100s. For
  long generations, use `stream: true`.
