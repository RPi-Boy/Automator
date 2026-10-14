"""Print (and save as summary.md) the headline table for one run of scripts/reproduce_jetson.sh.

  python3 scripts/summarize_jetson_run.py docs/jetson_results/run_YYYYmmdd-HHMMSS
Reads benchmark_*.json, trtexec_*_b*.log and sort_results.json from that folder; missing pieces are skipped.
"""
import glob
import json
import os
import re
import sys


def main():
    d = sys.argv[1]
    out = []
    mode = open(os.path.join(d, "power_mode.txt")).read().strip() if os.path.exists(os.path.join(d, "power_mode.txt")) else "unknown"
    out.append(f"Run folder: {d}")
    out.append(f"Power mode: {mode}")
    out.append("")
    out.append("Perception to plan latency per frame, 3 objects per frame (ms)")
    out.append(f"{'model / backend':56s} {'detect':>7s} {'network':>8s} {'plan':>6s} {'total mean':>11s} {'p95':>7s} {'max':>7s} {'iters':>6s}")
    for path in sorted(glob.glob(os.path.join(d, "benchmark_*.json"))):
        j = json.load(open(path))
        label = f"{os.path.basename(j['model'])} / {j['backend'].split(' ')[0]}"
        out.append(f"{label[:56]:56s} {j['detection']['mean_ms']:7.2f} {j['network']['mean_ms']:8.2f} "
                   f"{j['ik_and_planning']['mean_ms']:6.2f} {j['total']['mean_ms']:11.2f} {j['total']['p95_ms']:7.2f} "
                   f"{j['total']['max_ms']:7.2f} {j['iters']:6d}")
    for path in sorted(glob.glob(os.path.join(d, "trtexec_*_b*.log"))):
        txt = open(path, errors="ignore").read()
        m = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms, median = [\d.]+ ms.*?percentile\(95%\) = ([\d.]+)", txt)
        t = re.search(r"Throughput: ([\d.]+) qps", txt)
        name = os.path.basename(path)[len("trtexec_"):-len(".log")]
        if m:
            out.append("")
            out.append(f"TensorRT GPU compute only (trtexec {name}): mean {float(m.group(1)):.3f} ms, p95 {float(m.group(2)):.3f} ms"
                       + (f", throughput {float(t.group(1)):.0f} qps" if t else ""))
    sp = os.path.join(d, "sort_results.json")
    if os.path.exists(sp):
        r = json.load(open(sp))
        vids = sorted(glob.glob(os.path.join(d, "videos", "*.mp4")))
        out.append("")
        out.append(f"Sorting run on this device: {r['sorted_correctly']} / {r['known_objects_total']} objects in the correct bin "
                   f"({100 * r['success_rate']:.1f} %), trials ending in an e-stop: {r['estop_trials']}, videos: {len(vids)}")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
