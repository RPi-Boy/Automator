"""Single source of truth for geometry shared by the simulator, the sort demo and the ROS nodes.
Pure python (no PyBullet, no ROS), so every consumer imports the same numbers."""
import math

import numpy as np

TOOL_LENGTH = 0.09   # wrist (ee_link origin) -> fingertips, from the URDF

BIN_RADIUS = 0.33
BIN_ANGLES = (0.95, 1.45, 1.95, 2.45)  # class id (sphere, cube, cylinder, brick) -> bin direction (rad)
NUM_CLASSES = len(BIN_ANGLES)


def bin_position(class_id):
    a = BIN_ANGLES[class_id]
    return np.array([BIN_RADIUS * math.cos(a), BIN_RADIUS * math.sin(a), 0.03])


def bin_table():
    return {c: bin_position(c) for c in range(NUM_CLASSES)}
