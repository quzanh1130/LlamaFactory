#!/bin/bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
SHARE_FLAG="${SHARE_FLAG:-}"

echo "Starting MathVision demo..."
echo "  Host: ${HOST}"
echo "  Port: ${PORT}"
echo "  URL:  http://${HOST}:${PORT}"

if [[ -n "${SHARE_FLAG}" ]]; then
  python mathvision_gradio_app.py --server-name "${HOST}" --server-port "${PORT}" --share
else
  python mathvision_gradio_app.py --server-name "${HOST}" --server-port "${PORT}"
fi
