"""Train the shape classifier + xyz refiner + Mahalanobis OOD score on data/sim_shapes.npz.

Pipeline: train (augmented) -> switch to eval -> fit Mahalanobis on clean train features ->
calibrate the 0.75 threshold on held-out in-distribution validation crops -> evaluate on the
untouched test split and on unseen-object OOD scenes -> export ONNX.
Metrics are written to models/perception_metrics.json; nothing is reported that was not measured.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.mobilenet_uncertainty import DeployWrapper, MobileNetUncertaintyBackbone  # noqa: E402

SPLIT = {"train": 0, "val": 1, "test": 2, "ood_val": 3, "ood_test": 4}
CLASS_NAMES = ["sphere", "cube", "cylinder", "brick"]


def to_tensor(crops):
    return torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255.0


def augment(x, gen):
    """Photometric + small geometric jitter on a batch (B,3,H,W) in 0-1. No flips: offsets are world-frame."""
    B = x.shape[0]
    dev = x.device
    brightness = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.4
    contrast = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.4
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * brightness
    # random colour-channel permutation: colour carries no class information
    perm = torch.stack([torch.randperm(3, device=dev) for _ in range(B)])
    x = torch.gather(x, 1, perm.view(B, 3, 1, 1).expand_as(x))
    # small random scale/translate
    scale = 1 + (torch.rand(B, device=dev, generator=gen) - 0.5) * 0.16
    shift = (torch.rand(B, 2, device=dev, generator=gen) - 0.5) * 0.08
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, :, 2] = shift
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    x = x + torch.randn(x.shape, device=dev, generator=gen) * 0.02
    return x.clamp(0, 1)


@torch.no_grad()
def run_model(model, crops, bbox, device, bs=256):
    model.eval()
    logits, offs, zs = [], [], []
    for i in range(0, len(crops), bs):
        x = to_tensor(crops[i:i + bs]).to(device)
        b = torch.from_numpy(bbox[i:i + bs]).to(device)
        l, o, z = model(x, b)
        logits.append(l.cpu()); offs.append(o.cpu()); zs.append(z.cpu())
    return torch.cat(logits), torch.cat(offs), torch.cat(zs)


def auroc(pos_scores, neg_scores):
    """P(score_ood > score_in), ties counted half. Pure numpy."""
    s = np.concatenate([neg_scores, pos_scores])
    ranks = s.argsort().argsort() + 1.0
    # average ranks over ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n_neg, n_pos = len(neg_scores), len(pos_scores)
    return float((ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sim_shapes.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet-pretrained backbone (needs internet)")
    ap.add_argument("--calib-percentile", type=float, default=99.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/perception.pt")
    ap.add_argument("--onnx", default="models/perception.onnx")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data)
    sp = d["split"]
    sel = {k: np.where(sp == v)[0] for k, v in SPLIT.items()}
    crops, bbox, labels = d["crops"], d["bbox_feat"], d["labels"]
    resid = (d["true_xyz"] - d["geom_xyz"]).astype(np.float32)
    print(f"[train] device={device} samples: " + ", ".join(f"{k}={len(v)}" for k, v in sel.items()))

    model = MobileNetUncertaintyBackbone(num_classes=len(CLASS_NAMES), feature_dim=128,
                                         pretrained=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr = sel["train"]
    steps = args.epochs * ((len(tr) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    X = to_tensor(crops[tr]).to(device)
    B_ = torch.from_numpy(bbox[tr]).to(device)
    Y = torch.from_numpy(labels[tr]).to(device)
    R = torch.from_numpy(resid[tr]).to(device) / model.OFFSET_SCALE

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tr), device=device)
        tot = 0.0
        for i in range(0, len(tr), args.batch_size):
            idx = order[i:i + args.batch_size]
            if len(idx) < 2:
                continue
            x = augment(X[idx], gen)
            logits, offset, z = model(x, B_[idx])
            loss = F.cross_entropy(logits, Y[idx], label_smoothing=0.05) \
                + 0.5 * F.smooth_l1_loss(offset / model.OFFSET_SCALE, R[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch + 1:3d}/{args.epochs} loss {tot / len(tr):.4f}  ({time.time() - t0:.0f}s)")

    # --- statistics must come from the final weights in eval mode, on un-augmented data ---
    model.eval()
    _, _, z_tr = run_model(model, crops[tr], bbox[tr], device)
    model.fit_mahalanobis(z_tr.to(device), torch.from_numpy(labels[tr]).to(device))
    va = sel["val"]
    _, _, z_va = run_model(model, crops[va], bbox[va], device)
    model.calibrate(z_va.to(device), args.calib_percentile)

    # --- evaluation on untouched splits ---
    def evaluate(name):
        idx = sel[name]
        lg, off, z = run_model(model, crops[idx], bbox[idx], device)
        dist = model.compute_mahalanobis_distance(z.to(device)).cpu()
        return idx, lg, off, dist, model.uncertainty_from_distance(dist.to(device)).cpu()

    idx, lg, off, dist, unc = evaluate("test")
    pred = lg.argmax(1).numpy()
    acc = float((pred == labels[idx]).mean())
    conf = np.zeros((4, 4), int)
    for t, p_ in zip(labels[idx], pred):
        conf[t, p_] += 1
    err_geom = np.linalg.norm(resid[idx], axis=1) * 1000
    err_model = np.linalg.norm(resid[idx] - off.numpy(), axis=1) * 1000
    xy_geom = np.linalg.norm(resid[idx][:, :2], axis=1) * 1000
    xy_model = np.linalg.norm(resid[idx][:, :2] - off.numpy()[:, :2], axis=1) * 1000

    metrics = {
        "calibration": {"percentile": args.calib_percentile,
                        "distance_at_threshold": float(model.distance_at_threshold)},
        "test_id": {
            "n": int(len(idx)), "shape_accuracy": acc, "confusion": conf.tolist(),
            "xyz_error_mm_geometry_only": {"mean": float(err_geom.mean()), "p95": float(np.percentile(err_geom, 95))},
            "xyz_error_mm_with_refiner": {"mean": float(err_model.mean()), "p95": float(np.percentile(err_model, 95))},
            "xy_error_mm_geometry_only": float(xy_geom.mean()),
            "xy_error_mm_with_refiner": float(xy_model.mean()),
            "false_estop_rate_at_0.75": float((unc.numpy() > 0.75).mean()),
        },
    }
    for name in ("ood_val", "ood_test"):
        o_idx, _, _, o_dist, o_unc = evaluate(name)
        kinds = d["kind"][o_idx]
        m = {
            "n": int(len(o_idx)),
            "auroc_vs_test_id": auroc(o_dist.numpy(), dist.numpy()),
            "detection_rate_at_0.75": float((o_unc.numpy() > 0.75).mean()),
            "fpr_at_95_tpr": None,
        }
        thr95 = np.percentile(dist.numpy(), 95)  # threshold keeping 95% of in-dist
        m["fpr_at_95_tpr"] = float((o_dist.numpy() <= thr95).mean())  # OOD samples wrongly accepted
        for k, kname in ((1, "urdf_meshes"), (2, "capsule")):
            if (kinds == k).any():
                m[f"detection_rate_at_0.75_{kname}"] = float((o_unc.numpy()[kinds == k] > 0.75).mean())
                m[f"auroc_{kname}"] = auroc(o_dist.numpy()[kinds == k], dist.numpy())
        metrics[name] = m

    print(json.dumps(metrics, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "perception_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    export_onnx(model.cpu(), args.onnx)


def export_onnx(model, path):
    model.eval()
    wrapper = DeployWrapper(model).eval()  # export restores the wrapper's mode afterwards; keep it eval
    crop = torch.rand(1, 3, 128, 128)
    bbox = torch.rand(1, 4)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.onnx.export(
        wrapper, (crop, bbox), path, opset_version=17, dynamo=False, do_constant_folding=True,
        input_names=["crop", "bbox_feat"], output_names=["logits", "offset", "uncertainty"],
        dynamic_axes={k: {0: "batch"} for k in ("crop", "bbox_feat", "logits", "offset", "uncertainty")})
    print(f"[export] wrote {path}")


if __name__ == "__main__":
    main()
