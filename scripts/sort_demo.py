"""End-to-end shape sorting in simulation.

Fixed camera -> background-subtraction detector -> MobileNetV3 (class, xyz refinement, OOD score)
-> PickPlaceSequencer (analytic IK + quintic joint trajectories, the same code the Jetson's
joint_trajectory_node runs) -> PyBullet arm. Each shape goes to its own bin. Objects whose
uncertainty exceeds 0.75 are not touched and trigger the e-stop path instead.

  python scripts/sort_demo.py --trials 20 --objects 4
  python scripts/sort_demo.py --gui
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pybullet as p

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "arm_perception"))

from arm_perception.trajectory import PickPlaceSequencer  # noqa: E402
from perception.detector import detect  # noqa: E402
from perception.inference import load_backend  # noqa: E402
from sim.sim_arm import CTRL_HZ, PHYS_HZ, SimArm  # noqa: E402
from perception.targeting import choose_target  # noqa: E402
from sim.scene import BIN_VISUAL_RADIUS, CLASS_NAMES, NUM_CLASSES, SortScene, bin_table, in_pick_region  # noqa: E402



def spectator_frame(client, w=640, h=480):
    view = p.computeViewMatrix([0.85, 0.75, 0.55], [0.15, 0.12, 0.12], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(50, w / h, 0.05, 3.0)
    img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=client)[2]
    return np.reshape(img, (h, w, 4))[:, :, :3].astype(np.uint8)


def load_model(path):
    """Any backend the ROS node supports: .onnx (ONNX Runtime), .engine (TensorRT), .pt (PyTorch)."""
    return load_backend(path)


class LiveSink:
    """Writes the newest frame (atomically) and a metrics log so a web page can follow a run live."""

    def __init__(self, live_dir, jsonl=None, max_fps=8):
        self.dir, self.max_fps, self.last = live_dir, max_fps, 0.0
        if jsonl:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl)), exist_ok=True)
        self.jsonl = open(jsonl, "a", buffering=1) if jsonl else None
        if live_dir:
            os.makedirs(live_dir, exist_ok=True)

    def frame(self, rgb):
        now = time.time()
        if not self.dir or now - self.last < 1.0 / self.max_fps:
            return
        self.last = now
        import cv2
        tmp = os.path.join(self.dir, "live.tmp.jpg")
        cv2.imwrite(tmp, rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        os.replace(tmp, os.path.join(self.dir, "live.jpg"))

    def metric(self, **kw):
        if self.jsonl:
            self.jsonl.write(json.dumps({"t": time.time(), **kw}) + "\n")


def perceive(model, scene, reference):
    """Detect + classify. Timing of the last call is left in perceive.timing (ms)."""
    rgb, _ = scene.camera.render(scene.client)
    t0 = time.perf_counter()
    dets = [d for d in detect(rgb, reference, scene.camera) if in_pick_region(d["geom_xyz"][:2])]
    t1 = time.perf_counter()
    out = []
    if dets:
        crops = np.ascontiguousarray(np.stack([d["crop"] for d in dets]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        logits, off, unc = model(crops, np.stack([d["bbox_feat"] for d in dets]).astype(np.float32))
        for i, d in enumerate(dets):
            out.append({"xyz": d["geom_xyz"] + off[i], "cls": int(logits[i].argmax()), "unc": float(unc[i]), "bbox": d["bbox"]})
    t2 = time.perf_counter()
    perceive.timing = {"detect_ms": (t1 - t0) * 1e3, "network_ms": (t2 - t1) * 1e3, "objects": len(dets)}
    perceive.last_rgb = rgb
    return out


perceive.timing = {}
perceive.last_rgb = None


VISION_HZ = 5
SETTLE_S = 0.6        # same defaults as evidential_vision_node
MAX_SIM_S = 150.0


def run_trial(model, seed, n_objects, client, frames=None, n_unknown=0, live=None):
    """Replays the ROS dataflow in-process: control at CTRL_HZ with the node's default sequencer, vision at 5 Hz
    only while the arm is idle and settled (as evidential_vision_node does), worst-OOD-first target choice."""
    rng = np.random.default_rng(seed)
    p.resetSimulation(physicsClientId=client)
    p.setTimeStep(1.0 / PHYS_HZ, physicsClientId=client)
    scene = SortScene(client)
    scene.build_static()
    arm = SimArm(client)
    for _ in range(30):
        p.stepSimulation(physicsClientId=client)
    reference = scene.camera.render(client)[0]
    classes = [int(c) for c in rng.integers(0, NUM_CLASSES, n_objects)]
    for k in range(n_unknown):
        classes[k] = None   # unseen random mesh
    scene.spawn(classes, rng)
    start_pos = {o["body"]: np.array(o["pos"]) for o in scene.objects}
    unknown_bodies = [o["body"] for o in scene.objects if o["class_id"] is None]
    seq = PickPlaceSequencer()  # exactly how joint_trajectory_node builds it
    bins = bin_table()
    t0 = time.time()
    estops, notes = 0, []
    unknown_seen = unknown_flagged = 0
    unknown_in_region = sum(in_pick_region(start_pos[b][:2]) for b in unknown_bodies)
    dt = 1.0 / CTRL_HZ
    settle_ticks, vision_period = int(SETTLE_S * CTRL_HZ), CTRL_HZ // VISION_HZ
    idle_ticks, last_vision, empty, stuck, tick = settle_ticks, -vision_period, 0, 0, 0

    while tick < MAX_SIM_S * CTRL_HZ:
        arm.command(seq.tick(dt), scene.objects)
        tick += 1
        idle_ticks = 0 if seq.busy else idle_ticks + 1
        if seq.busy and tick % 3 == 0 and (frames is not None or live is not None):
            view = spectator_frame(client)
            if frames is not None:
                frames.append(view)
            if live is not None:
                live.frame(view)
        if seq.busy or idle_ticks < settle_ticks or tick - last_vision < vision_period:
            continue
        last_vision = tick
        dets = perceive(model, scene, reference)
        if live is not None:
            live.metric(kind="perception", seed=seed, **perceive.timing)
        for b in unknown_bodies:   # bookkeeping only: what did perception make of each unseen object?
            near = [d for d in dets if np.linalg.norm(d["xyz"][:2] - start_pos[b][:2]) < 0.05]
            if near and not arm.held:
                unknown_seen += 1
                unknown_flagged += int(max(d["unc"] for d in near) > seq.threshold)
        target = choose_target(dets, seq.threshold)
        if target is None:
            empty += 1
            if empty >= 3:
                break
            continue
        empty = 0
        if seq.on_target(target["xyz"], target["unc"], target["cls"]):
            continue
        if seq.halted:
            estops += 1
            on_unknown = any(np.linalg.norm(target["xyz"][:2] - start_pos[b][:2]) < 0.05 for b in unknown_bodies)
            notes.append(f"e-stop on {'UNKNOWN' if on_unknown else 'KNOWN'} object, class {CLASS_NAMES[target['cls']]} "
                         f"at {np.round(target['xyz'], 3).tolist()} u={target['unc']:.2f}")
            break
        stuck += 1               # unreachable target: the node would republish it forever
        if stuck >= 3:
            notes.append(f"stuck: {seq.last_reject}")
            break

    placed_ok = misplaced = left = unknown_touched = 0
    for o in scene.objects:
        pos = np.array(p.getBasePositionAndOrientation(o["body"], physicsClientId=client)[0])
        if o["class_id"] is None:
            unknown_touched += int(np.linalg.norm(pos[:2] - start_pos[o["body"]][:2]) > 0.02)
            continue
        if np.linalg.norm(pos[:2] - bins[o["class_id"]][:2]) < BIN_VISUAL_RADIUS + 0.01 and pos[2] < 0.1:
            placed_ok += 1
        elif in_pick_region(pos[:2]):
            left += 1
        else:
            misplaced += 1
            notes.append(f"{CLASS_NAMES[o['class_id']]} ended at {pos.round(3).tolist()} (bin {bins[o['class_id']][:2].round(3).tolist()})")
    return {"seed": seed, "objects": n_objects, "sorted_correctly": placed_ok, "wrong_or_off_bin": misplaced,
            "left_on_table": left, "estops": estops, "sim_seconds": time.time() - t0, "notes": notes,
            "unknown_objects": n_unknown, "unknown_touched": unknown_touched,
            "unknown_in_pick_region": unknown_in_region, "unknown_seen_by_vision": unknown_seen,
            "unknown_flagged": unknown_flagged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/perception.onnx", help=".onnx, .engine or .pt")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--objects", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--unknown", type=int, default=0, help="replace this many objects with unseen random meshes")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--video", default="", help="write an mp4 of the first trial from a spectator camera")
    ap.add_argument("--video-dir", default="", help="write one mp4 per trial into this directory")
    ap.add_argument("--live-dir", default="", help="keep live.jpg (newest frame) updated here, for the web dashboard")
    ap.add_argument("--metrics-jsonl", default="", help="append per-perception-cycle latency and per-trial results as JSON lines")
    ap.add_argument("--out", default="models/sort_results.json")
    args = ap.parse_args()

    model = load_model(os.path.join(ROOT, args.model))
    client = p.connect(p.GUI if args.gui else p.DIRECT)
    live = LiveSink(args.live_dir or None, args.metrics_jsonl or None)
    print(f"backend: {model.name}", flush=True)
    results = []
    for t in range(args.trials):
        want_video = bool(args.video_dir) or (args.video and t == 0)
        frames = [] if want_video else None
        r = run_trial(model, args.seed + t, args.objects, client, frames, args.unknown, live)
        if frames:
            import cv2
            path = os.path.join(args.video_dir, f"trial_{t:02d}_seed{args.seed + t}.mp4") if args.video_dir \
                else os.path.join(ROOT, args.video)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))
            for fr in frames:
                vw.write(fr[:, :, ::-1])
            vw.release()
            print(f'wrote {path} ({len(frames)} frames)', flush=True)
        live.metric(kind="trial", trial=t, sorted_correctly=r["sorted_correctly"], objects=r["objects"],
                    estops=r["estops"], sim_seconds=r["sim_seconds"])
        results.append(r)
        print(f"trial {t:2d}: {r['sorted_correctly']}/{r['objects']} sorted correctly, "
              f"{r['wrong_or_off_bin']} off-bin, {r['left_on_table']} left, e-stops {r['estops']}")
        for n in r["notes"]:
            print("      ", n)
    total = sum(r["objects"] - r["unknown_objects"] for r in results)
    ok = sum(r["sorted_correctly"] for r in results)
    summary = {"trials": args.trials, "known_objects_total": total, "sorted_correctly": ok,
               "success_rate": ok / max(total, 1), "estop_trials": sum(r["estops"] > 0 for r in results),
               "results": results}
    if args.unknown:
        touched = sum(r["unknown_touched"] for r in results)
        summary["unknown_objects_total"] = args.unknown * args.trials
        summary["unknown_objects_touched"] = touched
        summary["trials_with_estop"] = sum(r["estops"] > 0 for r in results)
        summary["unknown_in_pick_region"] = sum(r["unknown_in_pick_region"] for r in results)
        summary["trials_unknown_ever_seen"] = sum(r["unknown_seen_by_vision"] > 0 for r in results)
        summary["trials_unknown_ever_flagged"] = sum(r["unknown_flagged"] > 0 for r in results)
        summary["estop_on_unknown_trials"] = sum(any("UNKNOWN" in n for n in r["notes"]) for r in results)
        summary["estop_on_known_trials"] = sum(any("KNOWN object" in n and "UNKNOWN" not in n for n in r["notes"]) for r in results)
        print(f"UNKNOWN OBJECTS: e-stop in {summary['trials_with_estop']}/{args.trials} trials "
              f"({summary['estop_on_unknown_trials']} on the unknown, {summary['estop_on_known_trials']} on a known object); "
              f"unknown in pick region {summary['unknown_in_pick_region']}, seen by vision in {summary['trials_unknown_ever_seen']} trials, "
              f"flagged in {summary['trials_unknown_ever_flagged']}; {touched}/{summary['unknown_objects_total']} moved by the arm")
    print(f"SORT SUCCESS: {ok}/{total} known objects in the right bin = {100 * ok / max(total, 1):.1f}%; "
          f"trials ending in an e-stop: {summary['estop_trials']}/{args.trials}")
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
