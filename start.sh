#!/usr/bin/env bash
set -euo pipefail

: "${API_KEY:?API_KEY environment variable must be set}"

# RunPod convention: PUBLIC_KEY env → SSH access (handy for debugging/agents).
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  ssh-keygen -A >/dev/null
  /usr/sbin/sshd
fi

mkdir -p "$MODELS_DIR" "$LOCAL_RECIPES_DIR" "$HF_HOME"
cd /opt/model-pod/supervisor
exec /opt/supervisor-venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log
