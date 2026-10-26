#!/usr/bin/env bash
# Starts the same web dashboard that runs on the Jetson, but on this PC, so the tests can be started, watched and
# downloaded (logs, latency charts, sorting videos) from a browser. Uses the repo's .venv if there is one.
#
#   scripts/start_pc_dashboard.sh                 # reachable from other devices on your network, port 8080
#   scripts/start_pc_dashboard.sh --local         # only this PC (127.0.0.1)
#   scripts/start_pc_dashboard.sh --port 9000
#   scripts/start_pc_dashboard.sh --stop          # stop a dashboard started with this script
#   scripts/start_pc_dashboard.sh --bg            # run in the background; log in runs/dashboard.log
# The URL with the access token is printed. The token is kept in ~/.automator_dashboard_token (same URL every time).
# On a PC the page offers the ONNX Runtime (CPU) backend; the TensorRT jobs need a Jetson and are skipped here.
# Anyone with the URL can run the whitelisted tests on this PC, so use --local on an untrusted network.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=0.0.0.0; PORT=8080; BG=0; STOP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) HOST=127.0.0.1; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --bg) BG=1; shift ;;
    --stop) STOP=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
done
PIDFILE=runs/dashboard.pid
if [[ $STOP -eq 1 ]]; then
  if [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    kill "$(cat $PIDFILE)" && rm -f $PIDFILE && echo "dashboard stopped"
  else
    echo "no dashboard started by this script is running"; rm -f $PIDFILE
  fi
  exit 0
fi

PY=python3; [[ -x .venv/bin/python ]] && PY=.venv/bin/python
missing=$("$PY" -c "
import importlib.util
print(' '.join(m for m in ('numpy', 'cv2', 'pybullet', 'onnxruntime') if importlib.util.find_spec(m) is None))" 2>/dev/null || echo "python")
if [[ -n "$missing" ]]; then
  echo "Missing Python packages for $PY: $missing"
  echo "Install them with: $PY -m pip install -r requirements.txt"; exit 1
fi
command -v ffmpeg >/dev/null || echo "note: ffmpeg not found, sorting videos will download fine but cannot play inline in the browser"
if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
  echo "Port $PORT is already in use (is the dashboard already running? try --stop or --port N)"; exit 1
fi
mkdir -p runs
export AUTOMATOR_ROOT="$PWD"
if [[ $BG -eq 1 ]]; then
  nohup "$PY" webapp/server.py --host "$HOST" --port "$PORT" > runs/dashboard.log 2>&1 &
  echo $! > $PIDFILE
  sleep 1.5
  cat runs/dashboard.log
  echo "Running in the background (pid $(cat $PIDFILE)). Stop it with: scripts/start_pc_dashboard.sh --stop"
else
  exec "$PY" webapp/server.py --host "$HOST" --port "$PORT"
fi
