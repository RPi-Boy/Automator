"""PyBullet arm driven by [q1..q4, gripper] set-points, with a grasp weld at the fingertips."""
import os
import sys

import numpy as np
import pybullet as p

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOME_Q = np.zeros(4)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "arm_perception"))
from arm_perception.config import TOOL_LENGTH  # noqa: E402,F401

PHYS_HZ, CTRL_HZ = 240, 60


class SimArm:
    """PyBullet arm driven by [q1..q4, gripper] set-points, with a grasp constraint at the fingertips."""

    def __init__(self, client):
        self.c = client
        self.id = p.loadURDF(os.path.join(ROOT, "sim", "urdf", "arm_4dof.urdf"), useFixedBase=True, physicsClientId=client)
        self.joints, self.fingers, self.ee = [0, 1, 2, 3], [4, 5], 3
        self.constraint, self.held = None, None
        for j, q in zip(self.joints, HOME_Q):
            p.resetJointState(self.id, j, q, physicsClientId=client)
        for f in self.fingers:
            p.changeDynamics(self.id, f, lateralFriction=2.5, physicsClientId=client)

    def joint_positions(self):
        return np.array([p.getJointState(self.id, j, physicsClientId=self.c)[0] for j in self.joints])

    def fingertip(self):
        pos, orn = p.getLinkState(self.id, self.ee, physicsClientId=self.c)[4:6]
        return np.array(p.multiplyTransforms(pos, orn, [0, 0, TOOL_LENGTH], [0, 0, 0, 1])[0])

    def command(self, sp, objects):
        q, g = sp[:4], sp[4]
        p.setJointMotorControlArray(self.id, self.joints, p.POSITION_CONTROL, targetPositions=[float(v) for v in q],
                                    forces=[200.0] * 4, physicsClientId=self.c)
        for f, sign in zip(self.fingers, (1, -1)):
            p.setJointMotorControl2(self.id, f, p.POSITION_CONTROL, targetPosition=0.4 * g * sign, force=15.0,
                                    physicsClientId=self.c)
        if g < -0.2 and self.constraint is None:
            tip = self.fingertip()
            near = [(np.linalg.norm(np.array(p.getBasePositionAndOrientation(o["body"], physicsClientId=self.c)[0]) - tip), o)
                    for o in objects]
            d, obj = min(near, key=lambda t: t[0])
            if d < 0.045:
                self._grasp(obj)
        elif g > 0.2 and self.constraint is not None:
            p.removeConstraint(self.constraint, physicsClientId=self.c)
            self.constraint = self.held = None
        for _ in range(PHYS_HZ // CTRL_HZ):
            p.stepSimulation(physicsClientId=self.c)

    def _grasp(self, obj):
        # Keep the object exactly where it is relative to the wrist (no teleport) and weld it there.
        lpos, lorn = p.getLinkState(self.id, self.ee, physicsClientId=self.c)[4:6]
        opos, oorn = p.getBasePositionAndOrientation(obj["body"], physicsClientId=self.c)
        inv_p, inv_o = p.invertTransform(lpos, lorn)
        rel_p, rel_o = p.multiplyTransforms(inv_p, inv_o, opos, oorn)
        self.constraint = p.createConstraint(self.id, self.ee, obj["body"], -1, p.JOINT_FIXED, [0, 0, 0],
                                             rel_p, [0, 0, 0], rel_o, physicsClientId=self.c)
        self.held = obj
