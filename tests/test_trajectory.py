import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "arm_perception"))

from arm_perception.config import TOOL_LENGTH, bin_table  # noqa: E402
from arm_perception.kinematics import JOINT_LIMITS, JOINT_VEL_LIMITS, tool_tip  # noqa: E402
from arm_perception.trajectory import PickPlaceSequencer  # noqa: E402

DT = 0.02


class TestSequencer(unittest.TestCase):

    def run_pick(self, pick):
        seq = PickPlaceSequencer()
        self.assertTrue(seq.on_target(pick, 0.1), seq.last_reject)
        return seq, seq.sample_path(DT)

    def test_path_respects_limits_and_reaches_home(self):
        _, path = self.run_pick([0.30, 0.05, 0.03])
        q = path[:, :4]
        self.assertTrue(np.all(q >= JOINT_LIMITS[:, 0] - 1e-6) and np.all(q <= JOINT_LIMITS[:, 1] + 1e-6))
        vel = np.abs(np.diff(q, axis=0)) / DT
        self.assertTrue(np.all(vel.max(axis=0) <= JOINT_VEL_LIMITS), vel.max(axis=0))
        self.assertLess(np.abs(path[-1, :4]).max(), 1e-6)  # back at home
        self.assertEqual(path[-1, 4], 1.0)                 # gripper open

    def test_gripper_closes_at_grasp_and_opens_at_place(self):
        pick = np.array([0.30, 0.05, 0.03])
        seq, path = self.run_pick(pick)
        closed = path[:, 4] < -0.99
        self.assertTrue(closed.any())
        first, last = np.argmax(closed), len(closed) - 1 - np.argmax(closed[::-1])
        # Fully closed at the pick point, fully open again at the release point.
        self.assertLess(np.linalg.norm(tool_tip(path[first, :4], TOOL_LENGTH) - (pick + [0, 0, seq.grasp_dz])), 2e-3)
        rel = seq.place_xyz + [0, 0, seq.release_dz]
        self.assertLess(np.linalg.norm(tool_tip(path[last, :4], TOOL_LENGTH) - rel), 2e-3)

    def test_never_dips_below_floor(self):
        for pick in ([0.31, 0.0, 0.023], [0.34, 0.12, 0.023], [0.30, -0.14, 0.023], [0.36, 0.20, 0.023]):
            _, path = self.run_pick(pick)
            self.assertGreater(min(tool_tip(q[:4], TOOL_LENGTH)[2] for q in path), 0.02)   # fingertips stay above the floor

    def test_high_uncertainty_halts_and_blocks_until_cleared(self):
        seq = PickPlaceSequencer()
        self.assertFalse(seq.on_target([0.3, 0, 0.03], 0.9))
        self.assertTrue(seq.halted)
        self.assertFalse(seq.on_target([0.3, 0, 0.03], 0.1))
        seq.clear_halt()
        self.assertTrue(seq.on_target([0.3, 0, 0.03], 0.1))

    def test_unreachable_target_rejected_and_busy_blocks(self):
        seq = PickPlaceSequencer()
        self.assertFalse(seq.on_target([1.0, 0, 0.03], 0.1))
        self.assertTrue(seq.on_target([0.3, 0, 0.03], 0.1))
        self.assertFalse(seq.on_target([0.3, 0.1, 0.03], 0.1))
        self.assertEqual(seq.last_reject, "busy")

    def test_halt_mid_motion_freezes_setpoint(self):
        seq = PickPlaceSequencer()
        seq.on_target([0.3, 0, 0.03], 0.1)
        for _ in range(50):
            seq.tick(DT)
        before = seq.tick(0)
        seq.halt()
        self.assertTrue(np.allclose(seq.tick(DT), before))

    def test_node_defaults_sort_each_class_to_its_own_bin(self):
        """Built exactly like joint_trajectory_node builds it (no arguments): fingertips reach the object,
        and every class is released over its own bin. A class without an entry must not inherit another's bin."""
        bins = bin_table()
        pick = np.array([0.31, 0.06, 0.023])
        for cls in range(4):
            seq = PickPlaceSequencer()
            self.assertTrue(seq.on_target(pick, 0.1, cls), seq.last_reject)
            path = seq.sample_path(DT)
            closed = path[:, 4] < -0.99
            first, last = np.argmax(closed), len(closed) - 1 - np.argmax(closed[::-1])
            grasp_tip = tool_tip(path[first, :4], TOOL_LENGTH)
            self.assertLess(np.linalg.norm(grasp_tip[:2] - pick[:2]), 2e-3)                 # over the object
            self.assertLess(abs(grasp_tip[2] - (pick[2] + seq.grasp_dz)), 2e-3)             # at grasp height
            release_tip = tool_tip(path[last, :4], TOOL_LENGTH)
            self.assertLess(np.linalg.norm(release_tip[:2] - bins[cls][:2]), 2e-3, f"class {cls}")
        seq = PickPlaceSequencer()
        seq.on_target(pick, 0.1, 2)
        seq.sample_path(DT)
        self.assertTrue(seq.on_target(pick, 0.1, 7))          # unknown class id: falls back to place_xyz ...
        self.assertTrue(np.allclose(seq.place_by_class[2], bins[2]))   # ... and class 2's bin is untouched


if __name__ == "__main__":
    unittest.main()
