#!/usr/bin/env bash
# One-time dependency install on the Jetson (JetPack 6 / Ubuntu 22.04 with ROS 2 Humble already installed).
# Everything here was needed on a fresh JetPack 6.2 (L4T 36.5) board; each step is safe to re-run.
set -euo pipefail
sudo apt update
# Base tools, OpenCV/NumPy from apt, ffmpeg (browser-playable videos), micro-ROS agent, GL libs for pybullet.
sudo apt install -y python3-pip python3-numpy python3-opencv build-essential python3-dev cmake git ffmpeg \
  libgl1-mesa-dev libglu1-mesa-dev ros-humble-micro-ros-agent ros-humble-rqt-image-view || \
  echo "WARNING: some apt packages failed (is ros-humble-micro-ros-agent available? you can use docker instead)"
# TensorRT 10 + the CUDA runtime pieces it needs. A fresh image has NO TensorRT/CUDA. The DLA packages are not pulled
# in automatically, and without them TensorRT fails with "libnvdla_compiler.so / libcudla.so.1: cannot open shared object file".
sudo apt install -y tensorrt python3-libnvinfer libnvinfer-bin nvidia-l4t-dla-compiler libcudla-12-6
sudo ldconfig
# Python packages. numpy<2 is required: NumPy 2 breaks the apt OpenCV ("_ARRAY_API not found").
# pybullet may need compiling on ARM (10-20 min) if it is not installed. cuda-python is how the TensorRT backend talks to the GPU.
pip3 install --user "numpy<2" pybullet onnxruntime "cuda-python>=12.6,<13"
python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)"
echo "Done. Next: python3 scripts/jetson_selfcheck.py, then scripts/start_dashboard.sh"
