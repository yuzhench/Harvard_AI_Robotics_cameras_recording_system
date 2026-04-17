#!/bin/bash
# RealSense Recording Pipeline — start server
# Usage:
#   ./run.sh           # with real camera
#   ./run.sh --mock    # with mock camera (no hardware)

set -e

cd "$(dirname "$0")"

# Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "[INFO] Installing dependencies..."
  pip install -r requirements.txt
fi

# Parse --port from args purely for display; the real port is handled by
# backend.server's argparse via "$@".
PORT=8000
for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--port" ]]; then
    n=$((i+1)); PORT="${!n}"
  elif [[ "$arg" =~ ^--port=(.+)$ ]]; then
    PORT="${BASH_REMATCH[1]}"
  fi
done

echo "[INFO] Starting server at http://localhost:$PORT"
python3 -m backend.server "$@"
