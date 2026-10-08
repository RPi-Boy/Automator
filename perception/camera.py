"""Numpy-only pinhole camera (same conventions as PyBullet's computeViewMatrix / computeProjectionMatrixFOV),
so the Jetson perception node can back-project pixels without PyBullet installed."""
import math

import numpy as np

# The fixed sim camera: above and beside the pick region. Single source of truth for the simulator,
# the dataset generator and the Jetson perception node.
SIM_CAMERA = dict(eye=(0.15, 0.40, 0.45), target=(0.32, 0.0, 0.03), up=(0.0, 0.0, 1.0),
                  fov=55.0, width=640, height=480)


class PinholeCamera:
    def __init__(self, eye, target, up=(0.0, 0.0, 1.0), fov=55.0, width=640, height=480):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.up = np.asarray(up, float)
        self.fov, self.width, self.height = fov, width, height
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        s = np.cross(f, self.up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        # camera axes in world coordinates: x = s (right), y = u (up), z = -f (backwards)
        self.world_from_cam = np.eye(4)
        self.world_from_cam[:3, 0], self.world_from_cam[:3, 1], self.world_from_cam[:3, 2] = s, u, -f
        self.world_from_cam[:3, 3] = self.eye
        self.f = (height / 2.0) / math.tan(math.radians(fov) / 2.0)  # focal length in pixels (vertical fov)

    def pixel_to_world(self, u, v, plane_z):
        """Intersect the ray through pixel (u, v) with the horizontal plane z = plane_z."""
        ray_cam = np.array([(u - self.width / 2.0) / self.f, -(v - self.height / 2.0) / self.f, -1.0, 0.0])
        d = (self.world_from_cam @ ray_cam)[:3]
        if abs(d[2]) < 1e-9:
            raise ValueError("ray parallel to plane")
        return self.eye + d * ((plane_z - self.eye[2]) / d[2])

    def world_to_pixel(self, xyz):
        """Pixel (u, v) of a world point (inverse of pixel_to_world)."""
        p_cam = np.linalg.inv(self.world_from_cam) @ np.append(np.asarray(xyz, float), 1.0)
        return (self.width / 2.0 + self.f * p_cam[0] / -p_cam[2], self.height / 2.0 - self.f * p_cam[1] / -p_cam[2])

    def to_dict(self):
        return {"eye": self.eye.tolist(), "target": self.target.tolist(), "up": self.up.tolist(),
                "fov": self.fov, "width": self.width, "height": self.height}
