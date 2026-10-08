#!/usr/bin/env bash
# One command for the simulated demo on the Jetson.  Extra arguments are passed to ros2 launch, e.g.
#   scripts/run_jetson_demo.sh                                  # ONNX Runtime, window on
#   scripts/run_jetson_demo.sh model_path:=$PWD/models/perception_fp16.engine
#   scripts/run_jetson_demo.sh inject_unknown:=true             # safety test: unseen object on the table
#   scripts/run_jetson_demo.sh viewer:=false save_video:=      # headless
set -euo pipefail
cd "$(dirname "$0")/.."
export AUTOMATOR_ROOT="$PWD"
source /opt/ros/humble/setup.bash
[[ -f ros2_ws/install/setup.bash ]] || { echo "Not built yet. Run: (cd ros2_ws && colcon build --symlink-install --packages-select arm_perception)"; exit 1; }
source ros2_ws/install/setup.bash
exec ros2 launch arm_perception sort_demo.launch.py "$@"
