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
  if [ ! -f "$dest" ]; then
    echo "[qwen3.8-27b] Downloading $MODEL_REPO/$file -> $dest"
    if curl -fL "https://huggingface.co/$MODEL_REPO/resolve/main/$file" -o "$dest.part"; then
      mv "$dest.part" "$dest"
    else
      echo "[qwen3.8-27b] ERROR: failed to download $file" >&2
      rm -f "$dest.part"
      exit 1
    fi
  else
    echo "[qwen3.8-27b] Already present: $dest"
  fi
}

download_if_missing "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf" "$MODEL_FILE"
download_if_missing "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf" "$MMPROJ_FILE"

echo "[qwen3.8-27b] Models ready. Starting llama-server on :8000"

# NOTE: --spec-type/--reasoning-*/--n-gpu-layers flag names have moved around
# across recent llama.cpp releases (upstream MTP support is new, merged
# ~May 2026). Verify these against `llama-server --help` in this image's
# actual build (server-cuda13-b10524) before relying on this in production.
exec llama-server \
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
