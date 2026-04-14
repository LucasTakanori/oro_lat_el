#!/usr/bin/env bash
# Tongue ROM data collection — multi-operator launcher.
#
# Starts the FastAPI backend on 127.0.0.1:8000 and exposes it via
# Cloudflare Tunnel as a public HTTPS URL (required because browsers
# only allow camera access on HTTPS or localhost).
#
# Prereqs (one-time on the lab PC):
#   1. python venv with deps:  pip install -r requirements.txt
#   2. cloudflared installed:  see cloudflare.com/products/tunnel
#   3. Disable system sleep during collection sessions.
#
# The trycloudflare URL printed below is EPHEMERAL — it changes each run.
# For a stable URL, register a Cloudflare-managed domain (~$8/yr) and
# switch to a named tunnel. Out of scope for this build.
#
# If the university network blocks tunneling (rare), fall back to a hosted
# deploy: copy server.py + requirements.txt to a Fly.io / Render app and
# share that URL instead.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "error: cloudflared not found on PATH" >&2
  echo "install from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
  exit 1
fi

uvicorn server:app --host 127.0.0.1 --port 8000 &
UVI=$!
trap 'kill $UVI 2>/dev/null || true' EXIT

sleep 1
echo
echo "=== uvicorn running on http://127.0.0.1:8000 (pid $UVI) ==="
echo "=== starting cloudflared quick tunnel — copy the trycloudflare URL below ==="
echo

cloudflared tunnel --url http://127.0.0.1:8000
