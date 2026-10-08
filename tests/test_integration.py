import os
import sys
import tempfile
import unittest

import numpy as np
import pybullet as p
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from perception.detector import detect  # noqa: E402
from perception.mobilenet_uncertainty import DeployWrapper, MobileNetUncertaintyBackbone  # noqa: E402
from sim.scene import SortScene  # noqa: E402

MODEL_PT = os.path.join(ROOT, "models", "perception.pt")


class TestUncertaintyCalibration(unittest.TestCase):
    """The 0.75 e-stop threshold must mean something: ~1% of known data above it, far-away data above it."""

    def test_threshold_flags_ood_but_not_in_distribution(self):
        g = torch.Generator().manual_seed(0)
        D, C = 128, 4
        centers = torch.randn(C, D, generator=g) * 3
        def sample(n):
            y = torch.randint(0, C, (n,), generator=g)
            return centers[y] + torch.randn(n, D, generator=g), y
        model = MobileNetUncertaintyBackbone(num_classes=C, feature_dim=D)
        z, y = sample(4000)
        model.fit_mahalanobis(z, y)
        z_cal, _ = sample(2000)
        model.calibrate(z_cal, 99.0)

        z_test, _ = sample(2000)
        u_in = model.uncertainty_from_distance(model.compute_mahalanobis_distance(z_test))
        self.assertLess((u_in > 0.75).float().mean().item(), 0.03)          # false e-stops stay rare
        self.assertGreater((u_in < 0.75).float().mean().item(), 0.97)

        z_ood = torch.randn(500, D, generator=g) * 3 + 8                    # not near any class
        u_ood = model.uncertainty_from_distance(model.compute_mahalanobis_distance(z_ood))
        self.assertGreater((u_ood > 0.75).float().mean().item(), 0.99)
        self.assertLessEqual(u_ood.max().item(), 1.0)                       # contract: bounded [0, 1]


class TestOnnxParity(unittest.TestCase):
    def test_exported_graph_matches_torch(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")
        torch.manual_seed(0)
        model = MobileNetUncertaintyBackbone(num_classes=4, feature_dim=128)
        model.train()   # populate BatchNorm statistics so eval-mode outputs vary between samples
        for _ in range(3):
            model(torch.rand(16, 3, 128, 128))
        model.eval()
        crop = torch.stack([torch.rand(3, 128, 128) * s for s in (0.2, 0.6, 1.0)])
        bbox = torch.rand(3, 4)
        with torch.no_grad():
            z = model(crop, bbox)[2]
        model.fit_mahalanobis(z.repeat(40, 1) + torch.randn(120, 128) * 0.05, torch.arange(120) % 4)
        model.calibrate(z.repeat(10, 1) + torch.randn(30, 128) * 0.05, 99.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.onnx")
            wrapper = DeployWrapper(model).eval()
            torch.onnx.export(wrapper, (crop[:1], bbox[:1]), path, opset_version=17, dynamo=False,
                              input_names=["crop", "bbox_feat"], output_names=["logits", "offset", "uncertainty"],
                              dynamic_axes={k: {0: "b"} for k in ("crop", "bbox_feat", "logits", "offset", "uncertainty")})
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"crop": crop.numpy(), "bbox_feat": bbox.numpy()})   # batch 3 != export batch 1
        with torch.no_grad():
            ref = wrapper(crop, bbox)
        for a, b in zip(out, ref):
            np.testing.assert_allclose(a, b.numpy(), rtol=1e-3, atol=1e-4)
        self.assertGreater(np.abs(out[1][0] - out[1][2]).max(), 1e-4)   # rows really differ: batch handling is exercised


class TestDetectorGeometry(unittest.TestCase):
    def test_backprojection_matches_ground_truth(self):
        """Camera model check, independent of the CNN: with the true object height the blob centroid must
        back-project to the true xy; with the detector's fixed assumed height the error stays within 1.2 cm
        (the CNN residual head removes most of that)."""
        cid = p.connect(p.DIRECT)
        try:
            scene = SortScene(cid, with_bins=False)
            scene.build_static()
            ref, _ = scene.camera.render(cid)
            scene.spawn([0, 1, 2, 3], np.random.default_rng(3))
            rgb, _ = scene.camera.render(cid)
            dets = detect(rgb, ref, scene.camera)
            self.assertEqual(len(dets), 4)
            for d in dets:
                ys, xs = np.nonzero(d["mask"])
                cx, cy = xs.mean(), ys.mean()
                obj = min(scene.objects, key=lambda o: np.linalg.norm(scene.camera.pixel_to_world(cx, cy, o["pos"][2])[:2] - o["pos"][:2]))
                exact = scene.camera.pixel_to_world(cx, cy, obj["pos"][2])
                self.assertLess(np.linalg.norm(exact[:2] - obj["pos"][:2]), 0.006)
                self.assertLess(np.linalg.norm(d["geom_xyz"][:2] - obj["pos"][:2]), 0.012)
        finally:
            p.disconnect(cid)


@unittest.skipUnless(os.path.exists(MODEL_PT), "train first: scripts/train_perception.py")
class TestEndToEndSorting(unittest.TestCase):
    def test_sorts_three_objects_into_their_bins(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from sort_demo import load_model, run_trial
        model = load_model(MODEL_PT)
        cid = p.connect(p.DIRECT)
        try:
            results = [run_trial(model, 2000 + i, 3, cid) for i in range(3)]
        finally:
            p.disconnect(cid)
        sorted_ok = sum(r["sorted_correctly"] for r in results)
        print(f"\n[sort test] {sorted_ok}/9 objects in the right bin")
        self.assertGreaterEqual(sorted_ok, 7)


if __name__ == "__main__":
    unittest.main()
