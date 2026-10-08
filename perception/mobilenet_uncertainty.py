import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MobileNetUncertaintyBackbone(nn.Module):
    """MobileNetV3-small shape classifier with a Mahalanobis out-of-distribution score.

    Input is one 128x128 RGB object crop (float, 0-1, NCHW) plus optional bbox features
    [cx, cy, w, h] (normalised). Outputs:
      logits   - shape class
      offset   - (true_xyz - geometric_xyz) in metres: a residual refinement of the detector's
                 table-plane back-projection (the crop alone cannot know absolute size or position,
                 so it also sees the bbox features)
      z        - 128-d embedding used for the Mahalanobis score
    ImageNet normalisation lives inside the model so the ONNX/TensorRT engine takes raw 0-1 pixels.
    """

    OFFSET_SCALE = 0.01  # offset head works in units of 1 cm

    def __init__(self, num_classes=4, feature_dim=128, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            mobilenet = mobilenet_v3_small(weights=weights)
            self.backbone = mobilenet.features
            in_features = mobilenet.classifier[0].in_features
        except Exception:
            if pretrained:
                raise
            # Fallback lightweight CNN backbone if torchvision is unavailable
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.Hardswish(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.Hardswish(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.Hardswish(),
            )
            in_features = 128

        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_proj = nn.Linear(in_features, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.target_regressor = nn.Sequential(
            nn.Linear(feature_dim + 4, 64), nn.ReLU(), nn.Linear(64, 3))

        # Mahalanobis statistics, fitted after training in eval mode (see fit_mahalanobis)
        self.register_buffer("class_means", torch.zeros(num_classes, feature_dim))
        self.register_buffer("inv_covariance", torch.eye(feature_dim))
        # Raw distance that maps to uncertainty 0.75 (the e-stop threshold). Set by calibrate().
        self.register_buffer("distance_at_threshold", torch.tensor(1.0))
        self.is_fitted = False

    def extract_features(self, x):
        x = (x - self.img_mean) / self.img_std
        features = self.global_pool(self.backbone(x))
        return F.relu(self.feature_proj(torch.flatten(features, 1)))

    def forward(self, x, bbox_feat=None):
        z = self.extract_features(x)
        if bbox_feat is None:
            bbox_feat = torch.zeros(z.shape[0], 4, device=z.device, dtype=z.dtype)
        logits = self.classifier(z)
        offset = self.target_regressor(torch.cat([z, bbox_feat], dim=1)) * self.OFFSET_SCALE
        return logits, offset, z

    @torch.no_grad()
    def fit_mahalanobis(self, feature_list, class_labels):
        """Class-conditional means and a tied, ridge-regularised inverse covariance (call in eval mode)."""
        D = feature_list.shape[1]
        means = torch.zeros(self.num_classes, D, device=feature_list.device)
        centered = []
        for c in range(self.num_classes):
            mask = class_labels == c
            if mask.any():
                means[c] = feature_list[mask].mean(dim=0)
                centered.append(feature_list[mask] - means[c])
        self.class_means.copy_(means)
        if centered:
            all_c = torch.cat(centered, dim=0)
            cov = all_c.t() @ all_c / max(1, all_c.shape[0] - 1)
            cov += torch.eye(D, device=cov.device) * 1e-3 * cov.diagonal().mean()
            self.inv_covariance.copy_(torch.linalg.pinv(cov))
        self.is_fitted = True

    @torch.no_grad()
    def calibrate(self, in_dist_features, percentile=99.0):
        """Choose the raw distance that maps to uncertainty 0.75 as a percentile of held-out
        in-distribution distances, so a threshold of 0.75 flags (100 - percentile)% of known objects."""
        d = self.compute_mahalanobis_distance(in_dist_features)
        self.distance_at_threshold.copy_(torch.quantile(d, percentile / 100.0))

    def compute_mahalanobis_distance(self, z):
        """Minimum Mahalanobis distance to any class mean, batched. z: (B, D) -> (B,)."""
        diff = z.unsqueeze(1) - self.class_means.unsqueeze(0)          # (B, C, D)
        m = torch.einsum("bcd,de,bce->bc", diff, self.inv_covariance, diff)
        return torch.sqrt(torch.clamp(m, min=0.0)).min(dim=1).values

    def uncertainty_from_distance(self, d):
        """Calibrated score in [0, 1]; 0.75 <=> the calibrated distance threshold."""
        return torch.clamp(0.75 * d / self.distance_at_threshold, max=1.0)

    def predict_with_uncertainty(self, x, bbox_feat=None):
        self.eval()
        with torch.no_grad():
            logits, offset, z = self.forward(x, bbox_feat)
            class_ids = torch.argmax(logits, dim=1)
            uncertainty = self.uncertainty_from_distance(self.compute_mahalanobis_distance(z))
            return class_ids, offset, uncertainty


class DeployWrapper(nn.Module):
    """Single export graph: (crop, bbox_feat) -> (logits, offset, uncertainty). Mahalanobis is inside the
    graph, so TensorRT runs the whole perception step and the ROS node does no extra linear algebra."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, crop, bbox_feat):
        logits, offset, z = self.model(crop, bbox_feat)
        u = self.model.uncertainty_from_distance(self.model.compute_mahalanobis_distance(z))
        return logits, offset, u
