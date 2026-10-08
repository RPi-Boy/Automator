import unittest
import os
import sys
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.arm_env import ArmGraspEnv


class TestArmGraspEnv(unittest.TestCase):

    def setUp(self):
        self.env = ArmGraspEnv(render_mode=None, max_steps=50)

    def tearDown(self):
        self.env.close()

    def test_reset_space_shapes(self):
        obs, info = self.env.reset(seed=42)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (18,))
        self.assertIn("obj_type_id", info)

    def test_step_function(self):
        obs, info = self.env.reset(seed=42)
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(next_obs.shape, (18,))
        self.assertIsInstance(reward, (float, np.float32, np.float64))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)


if __name__ == "__main__":
    unittest.main()
