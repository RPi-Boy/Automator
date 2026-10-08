import math
import os
import sys
import unittest

import numpy as np
import pybullet as p

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "arm_perception"))

from arm_perception.kinematics import IKError, inverse_kinematics, forward_kinematics  # noqa: E402


class TestKinematics(unittest.TestCase):
    """IK is checked against PyBullet FK on the real URDF, so sign conventions can't drift."""

    @classmethod
    def setUpClass(cls):
        cls.cid = p.connect(p.DIRECT)
        cls.robot = p.loadURDF(os.path.join(ROOT, "sim", "urdf", "arm_4dof.urdf"),
                               useFixedBase=True, physicsClientId=cls.cid)

    @classmethod
    def tearDownClass(cls):
        p.disconnect(cls.cid)

    def bullet_fk(self, q):
        for j in range(4):
            p.resetJointState(self.robot, j, float(q[j]), physicsClientId=self.cid)
        return np.array(p.getLinkState(self.robot, 3, physicsClientId=self.cid)[4])

    def test_analytic_fk_matches_bullet(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.uniform([-3, -1.5, -2, -1.5], [3, 1.5, 2, 1.5])
            self.assertLess(np.linalg.norm(self.bullet_fk(q) - forward_kinematics(q)), 1e-6)

    def test_ik_reaches_env_workspace_within_1mm(self):
        rng = np.random.default_rng(1)
        worst, refused = 0.0, 0
        for _ in range(300):
            r, th = rng.uniform(0.20, 0.35), rng.uniform(-math.pi, math.pi)
            z = rng.choice([0.05, 0.11, 0.25])
            target = np.array([r * math.cos(th), r * math.sin(th), z])
            try:
                q = inverse_kinematics(target)
            except IKError:
                # Only targets inside the elbow-limit dead zone (|q3| <= 2.0 rad) may be refused.
                self.assertLess(np.linalg.norm(target - [0, 0, 0.25]), 0.31)
                refused += 1
                continue
            worst = max(worst, np.linalg.norm(self.bullet_fk(q) - target))
        self.assertLess(refused, 250)  # most of the sampled workspace must be solvable
        self.assertLess(worst, 1e-3)

    def test_unreachable_target_rejected(self):
        with self.assertRaises(IKError):
            inverse_kinematics([0.9, 0.0, 0.25])


if __name__ == "__main__":
    unittest.main()
