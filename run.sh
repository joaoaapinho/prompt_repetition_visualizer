#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
fi

echo "Installing / syncing dependencies..."
"$VENV/bin/pip" install -q -r "$SCRIPT_DIR/requirements.txt"

echo "Starting Prompt Repetition Visualizer at http://localhost:7860"
exec "$VENV/bin/python" "$SCRIPT_DIR/prompt_repetition_visualizer.py"
