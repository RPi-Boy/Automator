"""Render the sorting dataset from the fixed sim camera.

In-distribution: scenes of 1-4 objects drawn from the four sortable shapes (sphere, cube, cylinder,
brick) with random colours, sizes, positions, yaw and lighting. Colour is independent of class so it
cannot be used as a shortcut.
OOD: scenes of objects the classifier never sees - random procedural meshes from pybullet_data
(pool 0-499 for calibration/validation, 500-999 for the final test) plus a held-out capsule shape.
Each detected blob becomes one sample: 128x128 crop, bbox features, first-pass geometric xyz and the
ground-truth object centre (matched through the segmentation mask).
"""
import argparse
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.scene import NUM_CLASSES, OOD_CAPSULE, SortScene  # noqa: E402
from perception.detector import detect  # noqa: E402

LIGHTS = [(0.3, 0.2, 1.0), (-0.4, 0.3, 0.9), (0.6, -0.5, 0.8), (0.0, 0.0, 1.0),
          (-0.7, -0.2, 0.7), (0.2, 0.7, 0.8), (0.9, 0.1, 0.9), (-0.2, -0.8, 0.6)]
SPLIT = {"train": 0, "val": 1, "test": 2, "ood_val": 3, "ood_test": 4}


def make_jobs(n_id, n_ood, seed):
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_id):
        u = rng.random()
        split = "train" if u < 0.70 else "val" if u < 0.85 else "test"
        jobs.append((split, "id", int(rng.integers(1, 5)), int(rng.integers(1 << 30))))
    for i in range(n_ood):
        split = "ood_val" if i % 2 == 0 else "ood_test"
        kind = "capsule" if rng.random() < 0.25 else "urdf"
        jobs.append((split, kind, int(rng.integers(1, 4)), int(rng.integers(1 << 30))))
    return jobs


def worker(jobs):
    cid = p.connect(p.DIRECT)
    scene = SortScene(cid, with_bins=True)  # trays are in view at deployment, so they are in the data
    scene.build_static()
    refs = {}
    rows = []
    stats = Counter()
    for split, kind, n_obj, seed in jobs:
        rng = np.random.default_rng(seed)
        li = int(rng.integers(len(LIGHTS)))
        if li not in refs:
            refs[li] = scene.camera.render(cid, LIGHTS[li])[0]
        if kind == "id":
            classes = [int(c) for c in rng.integers(0, NUM_CLASSES, n_obj)]
        elif kind == "capsule":
            classes = [OOD_CAPSULE] * n_obj
        else:
            classes = [None] * n_obj
        try:
            if kind == "urdf":  # disjoint mesh pools for ood_val / ood_test
                import sim.scene as sc
                pool = range(0, 500) if split == "ood_val" else range(500, 1000)
                orig = sc.make_random_urdf
                sc.make_random_urdf = lambda c, r, max_dim=0.055, pool=pool: orig(c, r, max_dim, pool)
                try:
                    scene.spawn(classes, rng)
                finally:
                    sc.make_random_urdf = orig
            else:
                scene.spawn(classes, rng)
        except RuntimeError:
            continue
        rgb, seg = scene.camera.render(cid, LIGHTS[li])
        dets = detect(rgb, refs[li], scene.camera)
        body_to_obj = {o["body"]: o for o in scene.objects}
        seen = set()
        for d in dets:
            ids = Counter(seg[d["mask"]].tolist())
            body, count = ids.most_common(1)[0]
            if body not in body_to_obj or count / d["mask"].sum() < 0.7:
                stats["unmatched_blob"] += 1
                continue
            o = body_to_obj[body]
            seen.add(body)
            label = o["class_id"] if kind == "id" else -1
            rows.append((d["crop"], d["bbox_feat"], d["geom_xyz"].astype(np.float32), o["pos"].astype(np.float32),
                         label, SPLIT[split], 0 if kind == "id" else 1 if kind == "urdf" else 2))
        stats["objects"] += len(scene.objects)
        stats["detected"] += len(seen)
        scene.clear_objects()
    p.disconnect(cid)
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-id-scenes", type=int, default=2500)
    ap.add_argument("--n-ood-scenes", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sim_shapes.npz")
    args = ap.parse_args()

    jobs = make_jobs(args.n_id_scenes, args.n_ood_scenes, args.seed)
    chunks = [jobs[i::args.workers * 4] for i in range(args.workers * 4)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(worker, chunks)
    rows, stats = [], Counter()
    for r, s in results:
        rows += r
        stats.update(s)
    crops, bf, gx, tx, lab, split, kind = (np.array(x) for x in zip(*rows))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, crops=crops.astype(np.uint8), bbox_feat=bf.astype(np.float32),
                        geom_xyz=gx, true_xyz=tx, labels=lab.astype(np.int64),
                        split=split.astype(np.int64), kind=kind.astype(np.int64))
    print(f"[dataset] {len(rows)} samples in {time.time() - t0:.0f}s -> {args.out}")
    print(f"[dataset] detection recall {stats['detected']}/{stats['objects']}; unmatched/merged blobs {stats['unmatched_blob']}")
    for name, v in SPLIT.items():
        sel = split == v
        extra = f" class counts {np.bincount(lab[sel]).tolist()}" if name in ("train", "val", "test") else ""
        print(f"  {name:9s} {int(sel.sum()):6d}{extra}")


if __name__ == "__main__":
    main()
