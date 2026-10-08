import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data


class ArmGraspEnv(gym.Env):
    """
    Gymnasium PyBullet Environment for 4-DOF Manipulator with Angular 2-Finger Gripper.
    Supports multi-object Pick-and-Place manipulation across Cornell dataset object types.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, urdf_path=None, max_steps=100):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.urdf_path = urdf_path or os.path.join(
            os.path.dirname(__file__), "urdf", "arm_4dof.urdf"
        )

        # 5D Action Space: [dq1, dq2, dq3, dq4, dg_gripper]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # 18D Observation Space: [q1..q4, dq1..dq4, ee_xyz, obj_xyz, place_xyz, obj_type_id]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

        self.robot_id = None
        self.plane_id = None
        self.object_id = None
        self.target_receptacle_id = None
        self.cam_id = None

        self.ee_index = 3  # ee_link index
        self.finger_indices = [4, 5]  # left_finger_joint, right_finger_joint
        self.joint_indices = [0, 1, 2, 3]

        self.grasp_constraint_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.current_step = 0

        # Randomize object pick location
        if seed is not None:
            np.random.seed(seed)
        r_pick = np.random.uniform(0.20, 0.35)
        theta_pick = np.random.uniform(-0.5, 0.5)
        self.obj_pos = np.array([r_pick * np.cos(theta_pick), r_pick * np.sin(theta_pick), 0.03], dtype=np.float32)

        # Randomize target placement goal location
        r_place = np.random.uniform(0.20, 0.35)
        theta_place = np.random.uniform(1.0, 2.0)
        self.place_pos = np.array([r_place * np.cos(theta_place), r_place * np.sin(theta_place), 0.03], dtype=np.float32)

        # Object category ID (0: Sphere, 1: Cube, 2: Cylinder)
        self.obj_type_id = int(np.random.randint(0, 3))

        # Load 4-DOF Robot URDF
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.physics_client,
        )

        # Spawn dataset object at pick location
        if self.obj_type_id == 0:  # Sphere
            obj_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.025, rgbaColor=[0.9, 0.2, 0.2, 1.0], physicsClientId=self.physics_client)
            col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.025, physicsClientId=self.physics_client)
        elif self.obj_type_id == 1:  # Cube
            obj_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0.2, 0.8, 0.2, 1.0], physicsClientId=self.physics_client)
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], physicsClientId=self.physics_client)
        else:  # Cylinder
            obj_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.05, rgbaColor=[0.2, 0.2, 0.9, 1.0], physicsClientId=self.physics_client)
            col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.05, physicsClientId=self.physics_client)

        self.object_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=obj_shape,
            basePosition=self.obj_pos.tolist(),
            physicsClientId=self.physics_client,
        )

        # Spawn Placement Target Receptacle Cylinder at place location
        receptacle_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.01, rgbaColor=[1.0, 0.8, 0.0, 0.7], physicsClientId=self.physics_client)
        self.target_receptacle_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=receptacle_shape,
            basePosition=self.place_pos.tolist(),
            physicsClientId=self.physics_client,
        )

        # Create camera visual indicator at [0.0, 0.45, 0.55]
        self.cam_pos = [0.0, 0.45, 0.55]
        cam_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.02], rgbaColor=[0.0, 0.8, 0.9, 1.0], physicsClientId=self.physics_client)
        self.cam_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=cam_shape_id, basePosition=self.cam_pos, physicsClientId=self.physics_client)

        p.addUserDebugLine(self.cam_pos, self.obj_pos.tolist(), lineColorRGB=[1, 1, 0], lineWidth=3, physicsClientId=self.physics_client)

        # Increase contact friction properties to prevent slippage (mu = 2.5)
        p.changeDynamics(self.robot_id, self.finger_indices[0], lateralFriction=2.5, spinningFriction=1.0, physicsClientId=self.physics_client)
        p.changeDynamics(self.robot_id, self.finger_indices[1], lateralFriction=2.5, spinningFriction=1.0, physicsClientId=self.physics_client)
        p.changeDynamics(self.object_id, -1, lateralFriction=2.5, spinningFriction=1.0, physicsClientId=self.physics_client)

        # Reset joint angles to home pose
        for joint in self.joint_indices:
            p.resetJointState(self.robot_id, joint, 0.0, 0.0, physicsClientId=self.physics_client)

        # Open gripper fingers
        p.resetJointState(self.robot_id, self.finger_indices[0], 0.4, 0.0, physicsClientId=self.physics_client)
        p.resetJointState(self.robot_id, self.finger_indices[1], -0.4, 0.0, physicsClientId=self.physics_client)

        self.grasp_constraint_id = None
        p.stepSimulation(physicsClientId=self.physics_client)

        obs = self._get_observation()
        info = {"obj_type_id": self.obj_type_id}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        arm_action = action[:4]
        gripper_action = action[4]

        # Apply arm joint velocity control
        for i, joint in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(arm_action[i] * 2.0),
                force=10.0,
                physicsClientId=self.physics_client,
            )

        # Apply angular gripper finger control (+1.0 open, -1.0 close)
        left_angle = 0.4 * gripper_action
        right_angle = -0.4 * gripper_action
        p.setJointMotorControl2(self.robot_id, self.finger_indices[0], p.POSITION_CONTROL, targetPosition=left_angle, force=15.0, physicsClientId=self.physics_client)
        p.setJointMotorControl2(self.robot_id, self.finger_indices[1], p.POSITION_CONTROL, targetPosition=right_angle, force=15.0, physicsClientId=self.physics_client)

        p.stepSimulation(physicsClientId=self.physics_client)

        # Update object position observation
        obj_state = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.physics_client)
        curr_obj_pos = np.array(obj_state[0], dtype=np.float32)

        # Perform grasp attachment when fingers close around object
        ee_state = p.getLinkState(self.robot_id, self.ee_index, physicsClientId=self.physics_client)
        ee_pos = np.array(ee_state[0], dtype=np.float32)
        dist_ee_obj = np.linalg.norm(ee_pos - curr_obj_pos)

        if gripper_action < -0.2 and dist_ee_obj < 0.05 and self.grasp_constraint_id is None:
            self.grasp_constraint_id = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=self.ee_index,
                childBodyUniqueId=self.object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.03],
                childFramePosition=[0, 0, 0],
                physicsClientId=self.physics_client
            )
        elif gripper_action > 0.2 and self.grasp_constraint_id is not None:
            p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.physics_client)
            self.grasp_constraint_id = None

        self.current_step += 1
        obs = self._get_observation()

        # Compute Pick-and-Place reward
        dist_obj_place = np.linalg.norm(curr_obj_pos - self.place_pos)
        reward = -dist_obj_place

        terminated = bool(dist_obj_place < 0.03)
        truncated = bool(self.current_step >= self.max_steps)
        info = {"dist_to_place": dist_obj_place, "is_grasped": self.grasp_constraint_id is not None}

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        joint_states = [p.getJointState(self.robot_id, j, physicsClientId=self.physics_client) for j in self.joint_indices]
        q = [s[0] for s in joint_states]
        dq = [s[1] for s in joint_states]

        ee_state = p.getLinkState(self.robot_id, self.ee_index, physicsClientId=self.physics_client)
        ee_pos = ee_state[0]

        obj_state = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.physics_client)
        curr_obj_pos = obj_state[0]

        obs = np.concatenate([
            np.array(q, dtype=np.float32),
            np.array(dq, dtype=np.float32),
            np.array(ee_pos, dtype=np.float32),
            np.array(curr_obj_pos, dtype=np.float32),
            self.place_pos,
            np.array([float(self.obj_type_id)], dtype=np.float32)
        ])
        return obs

    def close(self):
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
