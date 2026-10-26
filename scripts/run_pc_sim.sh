#!/usr/bin/env bash
# Run the sorting simulation on this machine and save one browser-playable (H.264) video per trial.
#
#   scripts/run_pc_sim.sh                                   # 3 trials, seeds 1000-1002, 4 objects each
#   scripts/run_pc_sim.sh --seed 2000 --trials 5            # seeds 2000-2004
#   scripts/run_pc_sim.sh --seed 5000 --trials 3 --unknown 1   # one unseen object per scene: the arm should e-stop
#   scripts/run_pc_sim.sh --seed 42 --objects 6 --name my_test
# Options: --seed N (first seed; trial k uses seed N+k)  --trials N  --objects N (1-6)  --unknown N (0-2)
#          --model PATH (.onnx, .pt or .engine)  --name LABEL (output folder name)
# Output: runs/pc_sim_videos/<name>/trial_XX_seedNNNN.mp4 and <name>.json (results). runs/ is git-ignored.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=1000; TRIALS=3; OBJECTS=4; UNKNOWN=0; MODEL=models/perception.onnx; NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --objects) OBJECTS="$2"; shift 2 ;;
    --unknown) UNKNOWN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
done
PY=python3; [[ -x .venv/bin/python ]] && PY=.venv/bin/python
LAST=$((SEED + TRIALS - 1))
NAME="${NAME:-$([[ "$UNKNOWN" -gt 0 ]] && echo unseen_)seeds_${SEED}-${LAST}}"
OUT="runs/pc_sim_videos/$NAME"
mkdir -p "$OUT"
"$PY" scripts/sort_demo.py --model "$MODEL" --trials "$TRIALS" --objects "$OBJECTS" --unknown "$UNKNOWN" --seed "$SEED" \
  --video-dir "$OUT" --out "runs/pc_sim_videos/$NAME.json" 2>&1 | grep -v "build time"
if command -v ffmpeg >/dev/null; then   # OpenCV writes MPEG-4 part 2; H.264 plays everywhere and is about 7x smaller
  for v in "$OUT"/*.mp4; do
    ffmpeg -y -loglevel error -i "$v" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart "$v.tmp.mp4" && mv "$v.tmp.mp4" "$v"
  done
else
  echo "ffmpeg not found: videos are MPEG-4 part 2 (fine in VLC, not in browsers)"
fi
echo "Videos: $OUT"; ls -1 "$OUT"
