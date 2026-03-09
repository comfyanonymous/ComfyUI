#!/usr/bin/env bash
# Start the Studio server.
# Usage: ./start.sh [port] [comfyui-url]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export STUDIO_PORT="${1:-8189}"
export COMFYUI_URL="${2:-http://127.0.0.1:8188}"

echo "Starting Studio on http://localhost:${STUDIO_PORT}"
echo "ComfyUI expected at ${COMFYUI_URL}"
echo ""

cd "$SCRIPT_DIR"
python3 server.py
