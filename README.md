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
**Ollama**. Several models can run at once if VRAM allows. The supervisor
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

## CLI: `mp`

`mp` is installed on the pod, or run it locally from `supervisor/` with
`poetry run mp`. It reads `MP_URL` and `MP_API_KEY`, and every command exits
non-zero on failure.

```
mp status                         GPU / disk / URLs
mp recipes                        list recipes
mp push my-model.yaml             add or replace a runtime recipe
mp deploy <id> [--set k=v] --wait deploy and block until ready (prints connection info)
mp test <id>                      smoke suite: models, chat, stream + vision/tools/reasoning per capabilities
mp logs <id> [-f]                 engine / download logs
mp endpoints [<id>]               copy-pasteable env vars + curl
mp stop <id> [--keep-files]       stop; deletes the model files by default
mp restart <id>                   same params, keeps files
```

## Development (no GPU needed)

```bash
cd supervisor && poetry install
MP_DEV=1 API_KEY=dev MODELS_DIR=/tmp/mp/models LOCAL_RECIPES_DIR=/tmp/mp/recipes \
  poetry run uvicorn app.main:app --port 8000     # dev mode adds a fake echo engine
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
  libs. Pods need a CUDA 13 host driver, and `pod.sh` filters for one.
- The supervisor runs in its own venv so its dependencies never conflict with
  vLLM's.
- Downloads use `huggingface_hub` (hf_xet) in a killable child process with
  10 retries. Progress in the UI comes from the size of the download
  directory against the repo's file sizes.
- Set `HF_TOKEN` on the pod for gated repos and higher rate limits.
- RunPod's HTTP proxy times out non-streaming requests at about 100s. For
  long generations, use `stream: true`.
