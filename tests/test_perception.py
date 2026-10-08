import os
import sys
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.mobilenet_uncertainty import MobileNetUncertaintyBackbone


class TestMobileNetUncertainty(unittest.TestCase):

    def setUp(self):
        self.model = MobileNetUncertaintyBackbone(num_classes=5, feature_dim=128)
        self.model.eval()

    def test_forward_pass(self):
        x = torch.randn(2, 3, 224, 224)
        logits, target_xyz, z = self.model(x)

        self.assertEqual(logits.shape, (2, 5))
        self.assertEqual(target_xyz.shape, (2, 3))
        self.assertEqual(z.shape, (2, 128))

    def test_fit_and_compute_mahalanobis(self):
        feats = torch.randn(20, 128)
        labels = torch.randint(0, 5, (20,))
        self.model.fit_mahalanobis(feats, labels)
        self.assertTrue(self.model.is_fitted)

        test_x = torch.randn(2, 3, 224, 224)
        class_ids, target_xyz, uncertainty = self.model.predict_with_uncertainty(test_x)

        self.assertEqual(class_ids.shape, (2,))
        self.assertEqual(target_xyz.shape, (2, 3))
        self.assertEqual(uncertainty.shape, (2,))


if __name__ == "__main__":
    unittest.main()
