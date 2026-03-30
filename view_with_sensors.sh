#!/usr/bin/env bash
# Wrapper to run view_object_with_sensors.py with convenient flags
# Usage:
#   ./view_with_sensors.sh <ObjectName> [--save-sensors] [--save-images] [--log-interval N]
# Examples:
#   ./view_with_sensors.sh YcbBanana --save-sensors --log-interval 25
#   ./view_with_sensors.sh YcbHammer --save-sensors --save-images

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ObjectName> [--save-sensors] [--save-images] [--log-interval N]"
  echo "Examples:"
  echo "  $0 YcbBanana --save-sensors --log-interval 25"
  echo "  $0 YcbPear --save-sensors --save-images"
  exit 1
fi

OBJECT_NAME="$1"
shift || true

# Optional: set MuJoCo GL backend for Linux/headless environments
if [[ -z "${MUJOCO_GL:-}" ]]; then
  export MUJOCO_GL=egl
fi

# Prefer uv if available; fall back to python from current venv
if command -v uv >/dev/null 2>&1; then
  uv run python "$PROJECT_ROOT/view_object_with_sensors.py" "$OBJECT_NAME" "$@"
else
  python "$PROJECT_ROOT/view_object_with_sensors.py" "$OBJECT_NAME" "$@"
fi
