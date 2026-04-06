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

echo "[INFO] Starting server at http://localhost:8000"
python3 -m backend.server "$@"
