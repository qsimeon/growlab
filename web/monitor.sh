#!/bin/sh
# Keep web/data.json fresh while experiments run.
cd "$(dirname "$0")/.."
while true; do uv run python web/build_data.py >/dev/null 2>&1; sleep 15; done
