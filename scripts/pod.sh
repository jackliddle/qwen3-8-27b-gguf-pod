#!/usr/bin/env bash
# Create / inspect / terminate the model-pod GPU pod on RunPod (REST API v1).
#
#   scripts/pod.sh up [--gpu "NVIDIA A40,NVIDIA L40S"] [--disk 200] [--name model-pod] [--image ...]
#   scripts/pod.sh status      # pod state + URLs
#   scripts/pod.sh env         # print `export MP_URL=... MP_API_KEY=...` for the mp CLI
#   scripts/pod.sh down        # terminate (stops all billing; model files go with it)
#
# Needs RUNPOD_API_KEY. The pod ID + generated API key are kept in .pod (gitignored).
# Policy (from runpod-endpoints/ENDPOINTS.md): SECURE cloud only, recreate rather
# than resume, retry on "no instances available".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.pod"
API="https://rest.runpod.io/v1"
IMAGE="${IMAGE:-ghcr.io/jackliddle/model-pod:latest}"
# Cheapest-first fallback; all 48GB so the 27B Q8 recipe fits.
GPUS="${GPUS:-NVIDIA A40,NVIDIA RTX A6000,NVIDIA RTX 6000 Ada Generation,NVIDIA L40S}"
DISK="${DISK:-200}"
NAME="${NAME:-model-pod}"
RECIPES_REPO="${RECIPES_REPO:-jackliddle/qwen3-8-27b-gguf-pod}"

die() { echo "error: $*" >&2; exit 1; }
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"

rp() { # method path [json-body]
  local method="$1" path="$2" body="${3:-}"
  curl -sS -X "$method" "$API$path" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    ${body:+--data "$body"} -w '\n%{http_code}'
}

load_state() { [ -f "$STATE" ] || die "no pod recorded in $STATE (run: $0 up)"; source "$STATE"; }
proxy_url() { echo "https://${POD_ID}-8000.proxy.runpod.net"; }

print_urls() {
  local url; url="$(proxy_url)"
  cat <<INFO

Pod:          $POD_ID
Web UI:       $url/#key=$API_KEY
OpenAI base:  $url/v1
API key:      $API_KEY

  export MP_URL=$url MP_API_KEY=$API_KEY
INFO
}

cmd_up() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --gpu) GPUS="$2"; shift 2 ;;
      --disk) DISK="$2"; shift 2 ;;
      --name) NAME="$2"; shift 2 ;;
      --image) IMAGE="$2"; shift 2 ;;
      *) die "unknown option $1" ;;
    esac
  done
  if [ -f "$STATE" ]; then
    source "$STATE"
    die "pod $POD_ID already recorded in $STATE — '$0 down' first (or delete the file if it's gone)"
  fi

  API_KEY="$(openssl rand -hex 24)"
  body="$(GPUS="$GPUS" IMAGE="$IMAGE" DISK="$DISK" NAME="$NAME" API_KEY="$API_KEY" \
    RECIPES_REPO="$RECIPES_REPO" HF_TOKEN="${HF_TOKEN:-}" PUBLIC_KEY="${PUBLIC_KEY:-}" python3 - <<'PY'
import json, os
env = {"API_KEY": os.environ["API_KEY"], "RECIPES_REPO": os.environ["RECIPES_REPO"]}
for k in ("HF_TOKEN", "PUBLIC_KEY"):
    if os.environ.get(k):
        env[k] = os.environ[k]
print(json.dumps({
    "name": os.environ["NAME"],
    "imageName": os.environ["IMAGE"],
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuCount": 1,
    "gpuTypeIds": [g.strip() for g in os.environ["GPUS"].split(",")],
    "gpuTypePriority": "custom",
    # vLLM + llama.cpp in the image are CUDA 13 builds.
    "allowedCudaVersions": ["13.0", "13.1", "13.2", "13.3", "13.4"],
    "containerDiskInGb": int(os.environ["DISK"]),
    "volumeInGb": 0,
    "ports": ["8000/http", "22/tcp"],
    "env": env,
}))
PY
)"

  echo "Creating pod ($GPUS, ${DISK}GB disk, SECURE)…"
  for attempt in $(seq 1 20); do
    out="$(rp POST /pods "$body")"; code="${out##*$'\n'}"; resp="${out%$'\n'*}"
    if [ "$code" = 200 ] || [ "$code" = 201 ]; then break; fi
    echo "  attempt $attempt: HTTP $code: $(echo "$resp" | head -c 300)" >&2
    echo "$resp" | grep -qiE "no instances|not have the resources|unavailable|capacity" || die "create failed"
    sleep 15
  done
  [ "$code" = 200 ] || [ "$code" = 201 ] || die "gave up after 20 attempts"
  POD_ID="$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')"
  printf 'POD_ID=%q\nAPI_KEY=%q\n' "$POD_ID" "$API_KEY" > "$STATE"
  chmod 600 "$STATE"
  echo "Created $POD_ID ($(echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("machine",{}).get("gpuTypeId") or "", "$%s/hr" % d.get("costPerHr","?"))'))"

  echo "Waiting for the supervisor (first pull of the ~15GB image on a new host can take 10+ min)…"
  local url; url="$(proxy_url)"
  for _ in $(seq 1 180); do
    if curl -sf -m 5 "$url/api/health" >/dev/null 2>&1; then
      print_urls
      return
    fi
    sleep 10
  done
  echo "Supervisor not answering after 30 min; check the pod logs in the RunPod console." >&2
  print_urls
  exit 1
}

cmd_status() {
  load_state
  out="$(rp GET "/pods/$POD_ID")"; code="${out##*$'\n'}"; resp="${out%$'\n'*}"
  [ "$code" = 200 ] || die "HTTP $code: $resp"
  echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("Status:", d.get("desiredStatus"), "| GPU:", (d.get("machine") or {}).get("gpuTypeId"), "| $%s/hr" % d.get("costPerHr"))'
  if curl -sf -m 5 "$(proxy_url)/api/health" >/dev/null 2>&1; then echo "Supervisor: up"; else echo "Supervisor: not responding"; fi
  print_urls
}

cmd_env() { load_state; echo "export MP_URL=$(proxy_url) MP_API_KEY=$API_KEY"; }

cmd_down() {
  load_state
  out="$(rp DELETE "/pods/$POD_ID")"; code="${out##*$'\n'}"
  case "$code" in
    200|204) echo "Terminated $POD_ID" ;;
    404) echo "Pod $POD_ID already gone" ;;
    *) die "terminate failed: HTTP $code: ${out%$'\n'*}" ;;
  esac
  rm -f "$STATE"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  status) cmd_status ;;
  env) cmd_env ;;
  down) cmd_down ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
