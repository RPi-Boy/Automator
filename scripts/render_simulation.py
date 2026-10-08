import os
import sys
import numpy as np
from PIL import Image

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.arm_env import ArmGraspEnv
import pybullet as p


def render_simulation_scene(output_path="data/sim_render.png", artifact_path=None):
    """
    Renders 3D visual snapshot displaying the 4-DOF manipulator with Angular 2-Finger Gripper,
    dataset object (Cube/Sphere), camera visual indicator, and placement target cylinder.
    """
    print("[Simulation Renderer] Initializing PyBullet 3D scene renderer with Angular Gripper & Multi-Object Pick-and-Place...")
    env = ArmGraspEnv(render_mode=None)
    obs, info = env.reset(seed=101)  # Seed 101 loads Cube object

    # Overview 3D camera angle
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0.85, 0.85, 0.75],
        cameraTargetPosition=[0.0, 0.20, 0.25],
        cameraUpVector=[0, 0, 1],
        physicsClientId=env.physics_client
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=1.33,
        nearVal=0.1,
        farVal=3.0,
        physicsClientId=env.physics_client
    )

    obj_pos = obs[11:14]

    # Compute high-precision IK joint angles for object pick location
    ik_poses = p.calculateInverseKinematics(
        env.robot_id,
        env.ee_index,
        obj_pos,
        maxNumIterations=500,
        residualThreshold=1e-5,
        physicsClientId=env.physics_client
    )

    frames = []

    print("[Simulation Renderer] Simulating angular gripper pick-and-place trajectory...")
    for step in range(50):
        alpha = min(1.0, (step + 1) / 30.0)
        current_target_joints = [ik_poses[i] * alpha for i in range(4)]

        # Apply position control to arm joints and close angular gripper fingers around object
        p.setJointMotorControlArray(
            bodyUniqueId=env.robot_id,
            jointIndices=env.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=current_target_joints,
            forces=[100.0] * 4,
            physicsClientId=env.physics_client
        )

        gripper_action = -1.0 if step > 25 else 1.0
        p.setJointMotorControl2(env.robot_id, env.finger_indices[0], p.POSITION_CONTROL, targetPosition=0.4 * gripper_action, force=20.0, physicsClientId=env.physics_client)
        p.setJointMotorControl2(env.robot_id, env.finger_indices[1], p.POSITION_CONTROL, targetPosition=-0.4 * gripper_action, force=20.0, physicsClientId=env.physics_client)

        p.stepSimulation(physicsClientId=env.physics_client)

        if step in [0, 25, 49]:
            width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=env.physics_client
            )
            rgb_array = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
            frames.append(rgb_array)

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    final_frame = Image.fromarray(frames[-1])
    final_frame.save(output_path)
    print(f"[Simulation Renderer] Saved Angular Gripper 3D snapshot to '{output_path}'.")

    if artifact_path:
        os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
        final_frame.save(artifact_path)
        print(f"[Simulation Renderer] Saved artifact copy to '{artifact_path}'.")

    return True


if __name__ == "__main__":
    artifact_dir = "/home/werrewulf/.gemini/antigravity-ide/brain/16bbf462-968d-4e6b-ab2b-1a7f9eb10d8f"
    art_path = os.path.join(artifact_dir, "sim_render.png")
    render_simulation_scene(output_path="data/sim_render.png", artifact_path=art_path)
