#!/usr/bin/env bash
# Start the MOSS Image web UI (gallery, queue, progress, settings).
#   ./start_ui.sh
# Opens http://127.0.0.1:7877 in your browser. The MOSS backend on port 18765 is started
# on demand by the UI the first time you generate (or reuse it if it already runs).
set -euo pipefail
cd "$(dirname "$0")"

for f in moss_ui.py ui/index.html; do
  [[ -e "$f" ]] || { echo "error: $f missing" >&2; exit 1; }
done
[[ -x .venv/bin/python ]] || { echo "error: .venv missing - run ./setup.sh first" >&2; exit 1; }

if command -v open >/dev/null; then (sleep 1; open "http://127.0.0.1:${MOSS_UI_PORT:-7877}") & fi

exec .venv/bin/python moss_ui.py
