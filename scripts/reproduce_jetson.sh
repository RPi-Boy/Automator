#!/usr/bin/env bash
# One command to reproduce the on-device (Jetson) results: self-check, engines, latency benchmarks for ONNX Runtime CPU /
# TensorRT FP32 / TensorRT FP16, pure GPU time with trtexec, and a 3-trial x 4-object sorting run with video.
# Everything is written to docs/jetson_results/run_<timestamp>/ and a summary table is printed at the end.
#
#   scripts/reproduce_jetson.sh            # full run (500 benchmark iterations each, 3 sorting trials; about 15 minutes,
#                                          # most of it the CPU-bound simulator; add about 5 minutes if engines must be built)
#   scripts/reproduce_jetson.sh --quick    # smoke test: 50 iterations, 1 sorting trial
#
# Run from the repository root on the Jetson after scripts/jetson_install.sh. It never changes the power mode: for the numbers in
# docs/JETSON_RESULTS.md use MAXN_SUPER first:  sudo nvpmodel -m 2 && sudo jetson_clocks
set -euo pipefail
cd "$(dirname "$0")/.."
export AUTOMATOR_ROOT="$PWD"

ITERS=500; TRTEXEC_ITERS=300; TRIALS=3; QUICK=""
case "${1:-}" in
  "") ;;
  --quick) ITERS=50; TRTEXEC_ITERS=50; TRIALS=1; QUICK="_quick" ;;
  -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "unknown option: $1 (use --quick or --help)"; exit 2 ;;
esac

POWER="$( (nvpmodel -q 2>/dev/null || true) | sed -n 's/^NV Power Mode: *//p' | head -n1)"
POWER="${POWER:-unknown}"
echo "Power mode: $POWER"
if [[ "$POWER" != MAXN_SUPER* ]]; then
  echo "WARNING: power mode is not MAXN_SUPER. Results will be slower than docs/JETSON_RESULTS.md."
  echo "         Switch with: sudo nvpmodel -m 2 && sudo jetson_clocks   (continuing in 5 s, Ctrl+C to stop)"
  sleep 5
fi
MODE_TAG="$(echo "$POWER" | sed 's/ .*//; s/[^A-Za-z0-9]/_/g')"      # e.g. MAXN_SUPER or 25W
OUT="docs/jetson_results/run_$(date +%Y%m%d-%H%M%S)${QUICK}"
mkdir -p "$OUT"
echo "$POWER" > "$OUT/power_mode.txt"
echo "Writing results to $OUT"

echo; echo "== 1/6 self-check"
# --skip-ros: the benchmarks and the sorting demo do not need ROS, so missing ROS is only a warning here
python3 scripts/jetson_selfcheck.py --skip-ros | tee "$OUT/selfcheck.log"
cp jetson_selfcheck.txt "$OUT/jetson_selfcheck.txt"

echo; echo "== 2/6 TensorRT engines (built only if missing)"
for p in fp16 fp32; do
  if [[ -f "models/perception_$p.engine" ]]; then echo "models/perception_$p.engine exists, skipping"
  else scripts/build_trt_engine.sh "$p" 2>&1 | tee "$OUT/build_$p.log"; fi
done

echo; echo "== 3/6 latency benchmarks ($ITERS iterations each)"
bench() {  # bench <model> <tag>
  python3 scripts/benchmark_edge.py --model "$1" --iters "$ITERS" --tag "$2" | tee "$OUT/$2.log"
  cp "models/benchmark_$2.json" "$OUT/"
}
bench models/perception.onnx          "jetson_${MODE_TAG}_onnx_cpu"
bench models/perception_fp32.engine   "jetson_${MODE_TAG}_trt_fp32"
bench models/perception_fp16.engine   "jetson_${MODE_TAG}_trt_fp16"

echo; echo "== 4/6 pure GPU time, FP16 engine (trtexec, batch 3, $TRTEXEC_ITERS iterations)"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
"$TRTEXEC" --loadEngine=models/perception_fp16.engine --shapes=crop:3x3x128x128,bbox_feat:3x4 \
  --iterations="$TRTEXEC_ITERS" --warmUp=500 > "$OUT/trtexec_fp16_b3.log" 2>&1
grep -E "GPU Compute Time: min|Throughput" "$OUT/trtexec_fp16_b3.log" | sed 's/^.*\] \[I\] //'

echo; echo "== 5/6 sorting run, TensorRT FP16, $TRIALS trial(s) x 4 objects, seeds from 1000, with video"
python3 scripts/sort_demo.py --model models/perception_fp16.engine --trials "$TRIALS" --objects 4 --seed 1000 \
  --video-dir "$OUT/videos" --out "$OUT/sort_results.json" | tee "$OUT/sort.log"
if command -v ffmpeg >/dev/null; then   # OpenCV writes MPEG-4 part 2; H.264 plays in browsers and is about 10x smaller
  for v in "$OUT"/videos/*.mp4; do
    ffmpeg -y -loglevel error -i "$v" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart "$v.h264.mp4" \
      && mv "$v.h264.mp4" "$v"
  done
fi

echo; echo "== 6/6 summary"
python3 scripts/summarize_jetson_run.py "$OUT"
echo; echo "All files are in $OUT (summary.md is the table above)."
