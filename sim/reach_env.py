"""Fingertip reach task for SAC: drive the gripper tip to a random reachable target within 1 cm.

Targets are drawn from the same workspace the sorting pipeline uses (pick/place heights, full yaw)
and kept only if the analytic IK says the arm can reach them, so the success radius is attainable.
"""
import math
import os
import sys

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "arm_perception"))

from arm_perception.kinematics import JOINT_LIMITS, IKError, inverse_kinematics  # noqa: E402
from sim.sim_arm import TOOL_LENGTH  # noqa: E402


class ArmReachEnv(gym.Env):
    """obs (17): q1-4, dq1-4, fingertip xyz, target xyz, target - fingertip.  action (4): joint velocity in [-1, 1] (x 2 rad/s)."""

    def __init__(self, success_radius=0.01, max_steps=150, action_repeat=4, seed=None):
        super().__init__()
        self.success_radius, self.max_steps, self.action_repeat = success_radius, max_steps, action_repeat
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)
        self.client = p.connect(p.DIRECT)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        self.robot = p.loadURDF(os.path.join(ROOT, "sim", "urdf", "arm_4dof.urdf"), useFixedBase=True,
                                physicsClientId=self.client)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.target = np.zeros(3)

    def _sample_target(self):
        while True:
            r, th = self.rng.uniform(0.27, 0.36), self.rng.uniform(-math.pi, math.pi)
            t = np.array([r * math.cos(th), r * math.sin(th), self.rng.uniform(0.05, 0.30)])
            try:
                inverse_kinematics(t, tool_length=TOOL_LENGTH)
                return t
            except IKError:
                continue

    def _tip(self):
        pos, orn = p.getLinkState(self.robot, 3, physicsClientId=self.client)[4:6]
        return np.array(p.multiplyTransforms(pos, orn, [0, 0, TOOL_LENGTH], [0, 0, 0, 1])[0])

    def _obs(self):
        js = [p.getJointState(self.robot, j, physicsClientId=self.client) for j in range(4)]
        tip = self._tip()
        return np.concatenate([[s[0] for s in js], [s[1] for s in js], tip, self.target, self.target - tip]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        start = self.rng.uniform(JOINT_LIMITS[:, 0] * 0.3, JOINT_LIMITS[:, 1] * 0.3)
        for j in range(4):
            p.resetJointState(self.robot, j, float(start[j]), 0.0, physicsClientId=self.client)
        self.target = self._sample_target()
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        for j in range(4):
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, targetVelocity=float(action[j] * 2.0),
                                    force=50.0, physicsClientId=self.client)
        for _ in range(self.action_repeat):
            p.stepSimulation(physicsClientId=self.client)
        self.steps += 1
        obs = self._obs()
        dist = float(np.linalg.norm(obs[8:11] - self.target))
        success = dist < self.success_radius
        # dense -distance (x10 so SAC's entropy term stays small relative to it), a bonus inside 3x the radius
        # to guide the final approach, and a terminal bonus on success
        reward = -10.0 * dist - 0.01 * float(np.sum(action ** 2)) + (1.0 if dist < 3 * self.success_radius else 0.0) \
            + (20.0 if success else 0.0)
        return obs, reward, success, self.steps >= self.max_steps, {"dist": dist, "success": success}

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
