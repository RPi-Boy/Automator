#!/usr/bin/env bash
# Starts the micro-ROS agent for the ESP32 plugged in over USB.  Usage: scripts/start_microros_agent.sh [/dev/ttyUSB0]
# Start the agent AFTER the ESP32 is powered/flashed; if the board was already running, press its EN (reset) button once.
# Close the Arduino serial monitor first: only one program can hold the port.
set -euo pipefail
DEV="${1:-$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -n1 || true)}"
[[ -n "$DEV" ]] || { echo "No /dev/ttyUSB* or /dev/ttyACM* found. Plug in the ESP32 with a DATA USB cable (some cables are charge-only)."; exit 1; }
[[ -r "$DEV" && -w "$DEV" ]] || { echo "No permission on $DEV. Run: sudo usermod -aG dialout \$USER   then log out and back in."; exit 1; }
echo "Using $DEV @ 115200"
source /opt/ros/humble/setup.bash
if ros2 pkg prefix micro_ros_agent >/dev/null 2>&1; then
  exec ros2 run micro_ros_agent micro_ros_agent serial --dev "$DEV" -b 115200
elif command -v docker >/dev/null; then
  exec docker run -it --rm --net=host --privileged -v /dev:/dev microros/micro-ros-agent:humble serial --dev "$DEV" -b 115200
else
  echo "micro_ros_agent not found. Install:  sudo apt install ros-humble-micro-ros-agent   (or install docker)"; exit 1
fi
