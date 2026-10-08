"""Fixed-camera object detector: background subtraction -> blobs -> crops + table-plane back-projection.

Deliberately classical: the camera is fixed, so differencing against an empty-scene reference finds
objects of any shape (including unseen ones) without a detection network. The CNN then only has to
classify one centred crop, and the Mahalanobis score says whether that crop looks like anything it was
trained on. Works on numpy arrays only, so it runs unchanged in the sim and on the Jetson.
"""
import math

import cv2
import numpy as np

CROP_SIZE = 128
ASSUMED_CENTER_Z = 0.025  # plane used for the first-pass xyz; the CNN residual head corrects it


def detect_blobs(rgb, reference, diff_thresh=28, min_area=250, max_area=40000):
    """rgb, reference: HxWx3 uint8 (reference = same scene without objects).
    Returns [(mask, (x, y, w, h), (cx, cy))] sorted left-to-right for determinism."""
    c0, c1, c2 = cv2.split(cv2.absdiff(rgb, reference))
    diff = cv2.max(cv2.max(c0, c1), c2)
    binary = (diff > diff_thresh).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            blobs.append((labels == i, (int(x), int(y), int(w), int(h)), (float(cents[i][0]), float(cents[i][1]))))
    return sorted(blobs, key=lambda b: b[2][0])


def make_crop(rgb, mask, bbox, size=CROP_SIZE, margin=1.35):
    """Square crop centred on the blob. Everything outside the (slightly dilated) object mask is set to
    neutral gray, so trays, neighbouring objects and shadows cannot influence the classifier or the
    OOD score. Areas outside the image are gray too. Works on the crop window only (cheap)."""
    x, y, w, h = bbox
    side = int(max(w, h) * margin)
    x0, y0 = x + w // 2 - side // 2, y + h // 2 - side // 2
    H, W = rgb.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
    keep = cv2.dilate(mask[ya:yb, xa:xb].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    canvas = np.full((side, side, 3), 128, np.uint8)
    canvas[ya - y0:yb - y0, xa - x0:xb - x0] = np.where(keep[:, :, None], rgb[ya:yb, xa:xb], np.uint8(128))
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def bbox_features(bbox, img_w, img_h):
    """[cx, cy, w, h] normalised to the image: tells the residual head where the crop came from
    and how big the object is (the crop itself is rescaled and loses absolute size)."""
    x, y, w, h = bbox
    return np.array([(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h], dtype=np.float32)


def detect(rgb, reference, camera):
    """Full first-pass detection. Each result: crop, bbox feature vector, geometric xyz."""
    out = []
    for mask, bbox, (cx, cy) in detect_blobs(rgb, reference):
        out.append({
            "crop": make_crop(rgb, mask, bbox),
            "bbox": bbox,
            "bbox_feat": bbox_features(bbox, rgb.shape[1], rgb.shape[0]),
            "mask": mask,
            "geom_xyz": camera.pixel_to_world(cx, cy, ASSUMED_CENTER_Z),
        })
    return out


def in_pick_region(xy, r_range=(0.28, 0.41), theta_max=0.8):
    """True if a world xy lies in the pick area (so objects already sorted into bins are ignored)."""
    r, th = math.hypot(xy[0], xy[1]), math.atan2(xy[1], xy[0])
    return r_range[0] <= r <= r_range[1] and abs(th) <= theta_max
