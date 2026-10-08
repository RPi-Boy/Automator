#!/usr/bin/env bash
# Build the TensorRT engine ON the Jetson (engines are tied to the GPU, TensorRT version and JetPack,
# so one built on a PC will not load there). Usage: scripts/build_trt_engine.sh [fp16|fp32] [onnx] [engine]
set -euo pipefail
PREC="${1:-fp16}"
ONNX="${2:-models/perception.onnx}"
ENGINE="${3:-models/perception_${PREC}.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

FLAGS=()
[[ "$PREC" == "fp16" ]] && FLAGS+=(--fp16)

"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" "${FLAGS[@]}" \
  --minShapes=crop:1x3x128x128,bbox_feat:1x4 \
  --optShapes=crop:4x3x128x128,bbox_feat:4x4 \
  --maxShapes=crop:8x3x128x128,bbox_feat:8x4
echo "built $ENGINE"
