"""Analytic kinematics for the 4-DOF arm (numpy only, no PyBullet).

Geometry comes from sim/urdf/arm_4dof.urdf:
  joint_1 yaw about z at z=0.05, joint_2 pitch about y at +0.20 (shoulder z=0.25),
  link_2 = 0.30 m, link_3 = 0.25 m, joint_4 (wrist, = ee_link origin) pitch about y.
A pitch angle t about +y turns the link axis (0,0,1) into (sin t, 0, cos t), so the
planar arm lives in (radial, z) with direction (sin(cum_angle), cos(cum_angle)).
"""
import math
import numpy as np

SHOULDER_Z = 0.25
L1 = 0.30
L2 = 0.25
JOINT_LIMITS = np.array([
    [-math.pi, math.pi],
    [-1.5708, 1.5708],
    [-2.0, 2.0],
    [-1.5708, 1.5708],
])
JOINT_VEL_LIMITS = np.array([2.0, 2.0, 2.5, 3.0])  # rad/s, from the URDF
HOME_Q = np.zeros(4)


class IKError(ValueError):
    """Target is unreachable or violates a joint limit."""


def forward_kinematics(q):
    """Position of the ee_link origin (wrist) for joints q = [q1..q4]."""
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
    r = L1 * math.sin(q2) + L2 * math.sin(q2 + q3)
    z = SHOULDER_Z + L1 * math.cos(q2) + L2 * math.cos(q2 + q3)
    return np.array([r * math.cos(q1), r * math.sin(q1), z])


def tool_tip(q, tool_length):
    """Fingertip position: the wrist plus tool_length along the tool axis (pitch q2+q3+q4, yaw q1)."""
    phi = float(q[1] + q[2] + q[3])
    reach = tool_length * math.sin(phi)
    wrist = forward_kinematics(q)
    return wrist + np.array([reach * math.cos(q[0]), reach * math.sin(q[0]), tool_length * math.cos(phi)])


def inverse_kinematics(xyz, tool_pitch=math.pi, tool_length=0.0):
    """Joint angles [q1..q4] that put the tool point at xyz.

    tool_pitch is the desired cumulative pitch q2+q3+q4 of the tool axis
    (pi = pointing straight down); q4 is clamped to its limit, so the tool pitch is
    best-effort. With tool_length=0 xyz is the ee_link origin, which is the point
    sim/arm_env.py treats as the end effector. Raises IKError if unreachable.
    """
    x, y, z = (float(v) for v in xyz)
    q1 = math.atan2(y, x)
    r = math.hypot(x, y)
    # Back the wrist off the tool tip along the tool axis.
    r -= tool_length * math.sin(tool_pitch)
    dz = z - tool_length * math.cos(tool_pitch) - SHOULDER_Z

    d2 = r * r + dz * dz
    c3 = (d2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if abs(c3) > 1.0:
        raise IKError(f"target {xyz} out of reach (|cos q3|={abs(c3):.3f})")

    a = math.atan2(r, dz)  # angle of the shoulder->wrist line from vertical
    best = None
    for sign in (1.0, -1.0):  # elbow-forward first, then the mirrored solution
        q3 = sign * math.acos(c3)
        q2 = a - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
        q4 = min(max(tool_pitch - q2 - q3, JOINT_LIMITS[3, 0]), JOINT_LIMITS[3, 1])
        q = np.array([q1, q2, q3, q4])
        if np.all(q >= JOINT_LIMITS[:, 0] - 1e-9) and np.all(q <= JOINT_LIMITS[:, 1] + 1e-9):
            best = q
            break
    if best is None:
        raise IKError(f"target {xyz} violates joint limits")
    return best
