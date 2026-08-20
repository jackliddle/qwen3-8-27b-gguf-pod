# qwen3-8-27b-gguf-pod

Self-built Runpod Pod image serving
[`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF)
via `llama-server`, replacing the earlier `vllm/vllm-openai` deploy of the
stock `Qwen/Qwen3.8-27B` bf16 weights (see the parent `runpod-endpoints`
repo's `ENDPOINTS.md`).

- **Base image:** `ghcr.io/ggml-org/llama.cpp:server-cuda13-b10524` (pinned
  build tag — recheck for a newer `server-cuda13-*` tag before rebuilding
  from scratch; this one includes upstream MTP support,
  [ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673),
  merged ~May 2026).
- **Model:** `Q8_K_P` quant (31.5GB) + the BF16 vision projector (`mmproj`,
  0.93GB) — downloaded once at first container boot onto the pod's
  persistent volume at `/workspace/models` (skipped on subsequent boots if
  already present). Only these two files are fetched; the upstream repo
  also hosts 9 other quant variants and a "FastMTP" acceleration sidecar
  that requires a patched llama.cpp build — deliberately not used here (see
  `ENDPOINTS.md` for the supply-chain rationale). This build uses only the
  model's *embedded* MTP layers, which are stock-compatible with upstream
  llama.cpp.
- **API:** OpenAI-compatible `llama-server` on `:8000`, bearer-token
  protected via the `API_KEY` env var (required — the container refuses to
  start without it). Served model name aliased to `qwen3.8-27b` to match
  the existing API contract.
- Built and pushed to `ghcr.io/jackliddle/qwen3-8-27b-gguf-pod` by
  `.github/workflows/build.yml` on every push to `main`.

See the parent `runpod-endpoints` repo's `ENDPOINTS.md` for the deployed pod
ID, GPU, and proxy URL once live — **this repo only builds the image**;
Runpod template/pod creation is separate follow-up work.
