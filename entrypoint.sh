#!/usr/bin/env bash
set -uo pipefail

: "${API_KEY:?API_KEY environment variable must be set}"

# Only these two files are fetched (not the whole model repo, which lists
# 172GB across every quant variant plus the FastMTP sidecar we don't use).
MODEL_REPO="HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
MODELS_DIR="/workspace/models"
MODEL_FILE="$MODELS_DIR/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
MMPROJ_FILE="$MODELS_DIR/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"

mkdir -p "$MODELS_DIR"

download_if_missing() {
  local file="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "[qwen3.8-27b] Already present: $dest"
    return
  fi

  echo "[qwen3.8-27b] Downloading $MODEL_REPO/$file -> $dest"
  # hf download stages into HF's local cache and only materializes the file
  # at --local-dir once the transfer completes, so a partial/interrupted
  # attempt never leaves a corrupt file at $dest (no .part/mv dance needed).
  # HF_HUB_ENABLE_HF_TRANSFER=1 (set in the Dockerfile) switches this to
  # hf_transfer's chunked, multi-connection backend instead of a single
  # HTTP GET, which is what actually fixes the slow/stalling curl transfers.
  local attempt=1 max_attempts=10
  while [ "$attempt" -le "$max_attempts" ]; do
    echo "[qwen3.8-27b] Attempt $attempt/$max_attempts: $file"
    if hf download "$MODEL_REPO" "$file" --local-dir "$MODELS_DIR"; then
      echo "[qwen3.8-27b] Download complete: $dest"
      return
    fi
    echo "[qwen3.8-27b] Attempt $attempt failed"
    sleep 5
    attempt=$((attempt + 1))
  done

  echo "[qwen3.8-27b] ERROR: failed to download $file after $max_attempts attempts" >&2
  exit 1
}

download_if_missing "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf" "$MODEL_FILE"
download_if_missing "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf" "$MMPROJ_FILE"

echo "[qwen3.8-27b] Models ready. Starting llama-server on :8000"

# NOTE: --spec-type/--reasoning-*/--n-gpu-layers flag names have moved around
# across recent llama.cpp releases (upstream MTP support is new, merged
# ~May 2026). Verify these against `llama-server --help` in this image's
# actual build (server-cuda13-b10524) before relying on this in production.
#
# Full path required: the base image's "server" stage copies the binary to
# /app/llama-server and points its own ENTRYPOINT at that exact path, but
# never adds /app to $PATH, so a bare `exec llama-server` fails with
# "not found" (confirmed live 2026-09-01 — every earlier attempt had always
# died during the slow curl download, before ever reaching this line).
exec /app/llama-server \
  --model "$MODEL_FILE" \
  --mmproj "$MMPROJ_FILE" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$API_KEY" \
  --n-gpu-layers 999 \
  --flash-attn on \
  --ctx-size 32768 \
  --parallel 1 \
  --jinja \
  --reasoning on \
  --reasoning-effort medium \
  --reasoning-format deepseek \
  --reasoning-preserve \
  --spec-type draft-mtp \
  --alias qwen3.8-27b
