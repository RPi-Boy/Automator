#!/usr/bin/env bash
# One-time dependency install on the Jetson (JetPack 6 / Ubuntu 22.04 with ROS 2 Humble already installed).
set -euo pipefail
sudo apt update
sudo apt install -y python3-pip python3-numpy python3-opencv build-essential python3-dev cmake git \
  libgl1-mesa-dev libglu1-mesa-dev ros-humble-micro-ros-agent ros-humble-rqt-image-view || \
  echo "WARNING: some apt packages failed (is ros-humble-micro-ros-agent available? you can use docker instead)"
# pybullet has no ARM wheel: this compiles it and takes 10-20 minutes.
pip3 install --user pybullet onnxruntime
echo "Done. Next: run scripts/jetson_selfcheck.py"
