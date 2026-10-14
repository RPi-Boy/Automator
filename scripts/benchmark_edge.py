"""Per-stage latency of the perception -> planning path on whatever machine runs it (run it ON the Jetson).

Uses real frames from the sim camera. Stages: background-subtraction detection, network inference
(any backend: .engine / .onnx / .pt), IK + trajectory planning. Rendering is the simulator's cost, not the
robot's, so it is excluded. Writes models/benchmark_<tag>.json.

  python scripts/benchmark_edge.py --model models/perception_fp16.engine --tag jetson_fp16
  python scripts/benchmark_edge.py --model models/perception.onnx --tag laptop_onnx
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pybullet as p

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "arm_perception"))

from arm_perception.trajectory import PickPlaceSequencer  # noqa: E402
from perception.detector import detect, in_pick_region  # noqa: E402
from perception.inference import load_backend  # noqa: E402
from sim.scene import SortScene, bin_table  # noqa: E402


def stats(ms):
    a = np.array(ms)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/perception.onnx")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tag", default="local")
    ap.add_argument("--seconds", type=float, default=0, help="run for this long instead of --iters (live mode)")
    ap.add_argument("--hz", type=float, default=0, help="pace iterations at this rate (0 = flat out); latency excludes the wait")
    ap.add_argument("--jsonl", default="", help="append one JSON line per iteration (for the dashboard)")
    ap.add_argument("--live-dir", default="", help="keep live.jpg (annotated newest frame) updated here")
    ap.add_argument("--no-write", action="store_true", help="do not write models/benchmark_<tag>.json")
    args = ap.parse_args()

    backend = load_backend(os.path.join(ROOT, args.model))
    cid = p.connect(p.DIRECT)
    scene = SortScene(cid)
    scene.build_static()
    ref, _ = scene.camera.render(cid)
    frames = []
    rng = np.random.default_rng(0)
    for _ in range(args.frames):
        scene.clear_objects()
        scene.spawn([int(c) for c in rng.integers(0, 4, 3)], rng)
        frames.append(scene.camera.render(cid)[0])

    t_det, t_net, t_plan, t_total = [], [], [], []
    seq = PickPlaceSequencer()
    # warm-up (engine init, allocator, caches)
    for f in frames[:5]:
        ds = [d for d in detect(f, ref, scene.camera) if in_pick_region(d["geom_xyz"][:2])]
        crops = np.stack([d["crop"] for d in ds]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        backend(np.ascontiguousarray(crops), np.stack([d["bbox_feat"] for d in ds]))

    jl = None
    if args.jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.jsonl)), exist_ok=True)
        jl = open(args.jsonl, "a", buffering=1)
    if args.live_dir:
        os.makedirs(args.live_dir, exist_ok=True)
        from arm_perception.overlay import draw_detections
        import cv2
    deadline = time.time() + args.seconds if args.seconds else None
    last_live, i = 0.0, -1
    while True:
        i += 1
        if deadline is not None:
            if time.time() >= deadline:
                break
        elif i >= args.iters:
            break
        t_loop = time.perf_counter()
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        ds = [d for d in detect(f, ref, scene.camera) if in_pick_region(d["geom_xyz"][:2])]
        t1 = time.perf_counter()
        crops = np.ascontiguousarray(np.stack([d["crop"] for d in ds]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        logits, offset, unc = backend(crops, np.stack([d["bbox_feat"] for d in ds]))
        t2 = time.perf_counter()
        seq.halted = False
        seq._segments = []
        seq.on_target(ds[0]["geom_xyz"] + offset[0], float(unc[0]), int(logits[0].argmax()))
        t3 = time.perf_counter()
        t_det.append((t1 - t0) * 1e3); t_net.append((t2 - t1) * 1e3); t_plan.append((t3 - t2) * 1e3)
        t_total.append((t3 - t0) * 1e3)
        if jl:
            jl.write(json.dumps({"t": time.time(), "kind": "iter", "i": i, "detect_ms": t_det[-1], "network_ms": t_net[-1],
                                 "plan_ms": t_plan[-1], "total_ms": t_total[-1], "objects": len(ds)}) + "\n")
        if args.live_dir and time.time() - last_live > 0.2:
            last_live = time.time()
            res = [{"bbox": d["bbox"], "cls": int(logits[k].argmax()), "unc": float(unc[k])} for k, d in enumerate(ds)]
            img = draw_detections(f, res, 0.75, t_total[-1])
            tmp = os.path.join(args.live_dir, "live.tmp.jpg")
            cv2.imwrite(tmp, img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
            os.replace(tmp, os.path.join(args.live_dir, "live.jpg"))
        if args.hz:
            time.sleep(max(0.0, 1.0 / args.hz - (time.perf_counter() - t_loop)))

    result = {"machine": platform.platform(), "processor": platform.processor(), "backend": backend.name,
              "model": args.model, "iters": len(t_total), "objects_per_frame": 3,
              "detection": stats(t_det), "network": stats(t_net), "ik_and_planning": stats(t_plan),
              "total": stats(t_total)}
    print(json.dumps(result, indent=2))
    if jl:
        jl.write(json.dumps({"t": time.time(), "kind": "done", "summary": result}) + "\n")
    if not args.no_write:
        out = os.path.join(ROOT, "models", f"benchmark_{args.tag}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
