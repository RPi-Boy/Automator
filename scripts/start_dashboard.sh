#!/usr/bin/env bash
# Starts the web dashboard (live tests + downloads). Open the printed URL from any browser on the same network.
#   scripts/start_dashboard.sh            # foreground, port 8080
#   PORT=9000 scripts/start_dashboard.sh
cd "$(dirname "$0")/.."
export AUTOMATOR_ROOT="$PWD"
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
exec python3 webapp/server.py --port "${PORT:-8080}"
