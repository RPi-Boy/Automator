"""Shared sorting-scene definition: object shapes, bins, the fixed camera and rendering.

Used by the dataset generator, the end-to-end sort demo and ArmGraspEnv so the camera, the object
sizes and the bin layout cannot drift apart between training data and the running system.
"""
import colorsys
import math
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ros2_ws", "src", "arm_perception"))

from arm_perception.config import BIN_ANGLES, BIN_RADIUS, bin_position, bin_table  # noqa: E402,F401
from perception.camera import SIM_CAMERA, PinholeCamera  # noqa: E402
from perception.detector import in_pick_region  # noqa: E402,F401  (re-exported)

CLASS_NAMES = ["sphere", "cube", "cylinder", "brick"]
NUM_CLASSES = len(CLASS_NAMES)
OOD_CAPSULE = 99  # held-out primitive used only to test out-of-distribution detection

# Pick region and bins live where the URDF's joint limits allow IK (see kinematics.py: targets
# closer than ~0.30 m to the shoulder at (0, 0, 0.25) are unreachable with |q3| <= 2.0 rad).
PICK_R = (0.30, 0.38)
PICK_THETA = (-0.7, 0.7)
BIN_VISUAL_RADIUS = 0.065   # inner radius of the tray rim
MIN_OBJECT_SEPARATION = 0.11
# Silhouettes of objects closer than this in the image merge into one blob, which the OOD score (rightly) calls
# unfamiliar. The demo keeps spawned objects apart in image space; touching objects stopping the arm is accepted.
MIN_PIXEL_SEPARATION = 100

# Camera: fixed above and beside the pick region (replaces the old [0, 0.45, 0.55] marker, whose
# 17 px objects were too small to classify). Vertical FOV in degrees, PyBullet conventions.
CAM_EYE, CAM_TARGET, CAM_UP = SIM_CAMERA["eye"], SIM_CAMERA["target"], SIM_CAMERA["up"]
CAM_FOV, IMG_W, IMG_H = SIM_CAMERA["fov"], SIM_CAMERA["width"], SIM_CAMERA["height"]


class Camera(PinholeCamera):
    """PinholeCamera (geometry, shared with the Jetson node) plus PyBullet matrices for rendering."""

    def __init__(self, eye=CAM_EYE, target=CAM_TARGET, fov=CAM_FOV, width=IMG_W, height=IMG_H):
        super().__init__(eye, target, CAM_UP, fov, width, height)
        self.view = p.computeViewMatrix(list(eye), list(target), list(CAM_UP))
        self.proj = p.computeProjectionMatrixFOV(fov, width / height, 0.05, 3.0)

    def render(self, client, light_dir=(0.3, 0.2, 1.0), shadow=False):
        _, _, rgba, _, seg = p.getCameraImage(
            self.width, self.height, self.view, self.proj, lightDirection=list(light_dir),
            shadow=int(shadow), renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=client)
        rgb = np.reshape(rgba, (self.height, self.width, 4))[:, :, :3].astype(np.uint8)
        return rgb, np.reshape(seg, (self.height, self.width)) & ((1 << 24) - 1)


def random_color(rng):
    h, s, v = rng.uniform(0, 1), rng.uniform(0.45, 1.0), rng.uniform(0.55, 1.0)
    return list(colorsys.hsv_to_rgb(h, s, v)) + [1.0]


def make_primitive(client, class_id, rng, mass=0.05, color=None):
    """Spawn one sorting object resting on z=0. Returns (body_id, center_height). Sizes are
    randomised within +-15% so size cannot stand in for shape."""
    color = color or random_color(rng)
    if class_id == 0:
        r = rng.uniform(0.022, 0.028)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color, physicsClientId=client)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=client)
        half_h = r
    elif class_id == 1:
        h = rng.uniform(0.017, 0.023)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[h] * 3, rgbaColor=color, physicsClientId=client)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h] * 3, physicsClientId=client)
        half_h = h
    elif class_id == 2:
        r, length = rng.uniform(0.017, 0.023), rng.uniform(0.045, 0.055)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=length, rgbaColor=color, physicsClientId=client)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=length, physicsClientId=client)
        half_h = length / 2
    elif class_id == 3:
        he = [rng.uniform(0.032, 0.040), rng.uniform(0.016, 0.020), rng.uniform(0.016, 0.020)]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=color, physicsClientId=client)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he, physicsClientId=client)
        half_h = he[2]
    elif class_id == OOD_CAPSULE:  # held-out shape: never in the training classes
        r, length = rng.uniform(0.016, 0.02), rng.uniform(0.04, 0.05)
        vis = p.createVisualShape(p.GEOM_CAPSULE, radius=r, length=length, rgbaColor=color, physicsClientId=client)
        col = p.createCollisionShape(p.GEOM_CAPSULE, radius=r, height=length, physicsClientId=client)
        half_h = length / 2 + r
    else:
        raise ValueError(class_id)
    yaw = rng.uniform(-math.pi, math.pi)
    body = p.createMultiBody(mass, col, vis, [0, 0, half_h],
                             p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=client)
    # PyBullet has no rolling resistance by default: without these a tipped cylinder rolls forever.
    p.changeDynamics(body, -1, lateralFriction=1.5, spinningFriction=0.5, rollingFriction=0.02,
                     linearDamping=0.3, angularDamping=0.6, physicsClientId=client)
    return body, half_h


def make_random_urdf(client, rng, max_dim=0.055, pool=range(1000)):
    """Unseen-object spawner (OOD): a random procedural mesh from pybullet_data, scaled to max_dim,
    dropped and settled. Returns (body_id, center_height)."""
    idx = int(rng.choice(list(pool)))
    path = os.path.join(pybullet_data.getDataPath(), "random_urdfs", f"{idx:03d}", f"{idx:03d}.urdf")
    body = p.loadURDF(path, [0, 0, 0.3], globalScaling=1.0, physicsClientId=client)
    lo, hi = p.getAABB(body, physicsClientId=client)
    scale = max_dim / max(np.array(hi) - np.array(lo))
    p.removeBody(body, physicsClientId=client)
    body = p.loadURDF(path, [0, 0, 0.1], globalScaling=float(scale), physicsClientId=client)
    return body, None


def sample_positions(rng, n, min_sep=MIN_OBJECT_SEPARATION, restarts=2000, camera=None, min_px=0):
    """Rejection-sample n pick positions at least min_sep apart (restarts from scratch when a greedy
    draw paints itself into a corner)."""
    for _ in range(restarts):
        pts = []
        for _ in range(60):
            r, th = rng.uniform(*PICK_R), rng.uniform(*PICK_THETA)
            c = np.array([r * math.cos(th), r * math.sin(th)])
            ok = all(np.linalg.norm(c - q) >= min_sep for q in pts)
            if ok and camera is not None and min_px:
                pc = np.array(camera.world_to_pixel([c[0], c[1], 0.025]))
                ok = all(np.linalg.norm(pc - np.array(camera.world_to_pixel([q[0], q[1], 0.025]))) >= min_px for q in pts)
            if ok:
                pts.append(c)
            if len(pts) == n:
                return pts
    raise RuntimeError("could not place objects with the requested separation")


class SortScene:
    """Floor + bin markers + a set of objects, inside an existing PyBullet client."""

    def __init__(self, client, camera=None, with_bins=True):
        self.client = client
        self.camera = camera or Camera()
        self.with_bins = with_bins
        self.objects = []  # dicts: body, class_id (None for OOD), pos
        self._static = []

    def build_static(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        if self.with_bins:
            for c in range(NUM_CLASSES):
                self._static += self._make_tray(bin_position(c))

    def _make_tray(self, center, n_segments=16, height=0.025, thickness=0.006):
        """Shallow ring of boxes: a rolling object released inside cannot leave the bin."""
        ids = []
        seg_len = 2 * math.pi * (BIN_VISUAL_RADIUS + thickness) / n_segments * 0.55
        for k in range(n_segments):
            a = 2 * math.pi * k / n_segments
            pos = [center[0] + (BIN_VISUAL_RADIUS + thickness / 2) * math.cos(a),
                   center[1] + (BIN_VISUAL_RADIUS + thickness / 2) * math.sin(a), height / 2]
            orn = p.getQuaternionFromEuler([0, 0, a])
            half = [thickness / 2, seg_len, height / 2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.95, 0.8, 0.1, 1], physicsClientId=self.client)
            ids.append(p.createMultiBody(0, col, vis, pos, orn, physicsClientId=self.client))
        return ids

    def clear_objects(self):
        for o in self.objects:
            p.removeBody(o["body"], physicsClientId=self.client)
        self.objects = []

    def spawn(self, class_ids, rng, positions=None):
        """class_ids entries: int shape id, or None for an unseen random URDF (OOD)."""
        positions = positions or sample_positions(rng, len(class_ids), camera=self.camera, min_px=MIN_PIXEL_SEPARATION)
        for cid, xy in zip(class_ids, positions):
            if cid is None:
                body, half_h = make_random_urdf(self.client, rng)
                p.resetBasePositionAndOrientation(body, [xy[0], xy[1], 0.1], [0, 0, 0, 1], physicsClientId=self.client)
            else:
                body, half_h = make_primitive(self.client, cid, rng)
                p.resetBasePositionAndOrientation(
                    body, [xy[0], xy[1], half_h], p.getBasePositionAndOrientation(body, physicsClientId=self.client)[1],
                    physicsClientId=self.client)
            self.objects.append({"body": body, "class_id": cid, "half_h": half_h})
        if any(o["class_id"] is None for o in self.objects):
            for _ in range(120):  # let unseen meshes settle
                p.stepSimulation(physicsClientId=self.client)
        for o in self.objects:
            o["pos"] = np.array(p.getBasePositionAndOrientation(o["body"], physicsClientId=self.client)[0])

    def object_positions(self):
        return [np.array(p.getBasePositionAndOrientation(o["body"], physicsClientId=self.client)[0])
                for o in self.objects]
