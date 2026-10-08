import os
import sys
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.arm_env import ArmGraspEnv
import pybullet as p


def record_simulation_video(mp4_output="data/sim_trajectory.mp4", webp_output="data/sim_trajectory.webp", artifact_dir=None):
    """
    Captures PyBullet simulation frames displaying the 4-DOF manipulator with Angular 2-Finger Gripper
    executing a full Pick -> Grasp -> Transport -> Place workflow on Cornell dataset objects.
    """
    print("[Video Recorder] Initializing PyBullet 3D simulation video recorder (Angular Gripper Pick-and-Place)...")
    env = ArmGraspEnv(render_mode=None)
    obs, info = env.reset(seed=101)  # Seed 101 spawns Cube object

    width, height = 640, 480
    fps = 30
    total_frames = 150

    obj_pos = obs[11:14]
    place_pos = obs[14:17]

    # Phase 1 IK: Pick Location (Above object)
    pick_ik_poses = p.calculateInverseKinematics(
        env.robot_id, env.ee_index, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.02],
        maxNumIterations=500, residualThreshold=1e-5, physicsClientId=env.physics_client
    )

    # Phase 2 IK: Lift Location
    lift_ik_poses = p.calculateInverseKinematics(
        env.robot_id, env.ee_index, [obj_pos[0], obj_pos[1], 0.25],
        maxNumIterations=500, residualThreshold=1e-5, physicsClientId=env.physics_client
    )

    # Phase 3 IK: Place Location (Target Receptacle)
    place_ik_poses = p.calculateInverseKinematics(
        env.robot_id, env.ee_index, [place_pos[0], place_pos[1], 0.08],
        maxNumIterations=500, residualThreshold=1e-5, physicsClientId=env.physics_client
    )

    frames_bgr = []
    pil_images = []

    print(f"[Video Recorder] Executing 150-frame Pick -> Grasp -> Transport -> Place manipulation trajectory...")

    for step in range(total_frames):
        # 4-Phase Trajectory Execution
        if step < 40:
            # Phase 1: Approach & Open Gripper
            t = (step + 1) / 40.0
            target_q = [pick_ik_poses[i] * t for i in range(4)]
            gripper_action = 1.0  # Open
        elif step < 60:
            # Phase 2: Lower & Grasp Object
            target_q = list(pick_ik_poses[:4])
            gripper_action = -1.0  # Close angular gripper
        elif step < 110:
            # Phase 3: Lift & Transport to Place Receptacle
            t = (step - 60) / 50.0
            target_q = [(1 - t) * lift_ik_poses[i] + t * place_ik_poses[i] for i in range(4)]
            gripper_action = -1.0  # Hold grasp
        else:
            # Phase 4: Place & Release Object
            target_q = list(place_ik_poses[:4])
            gripper_action = 1.0  # Open to release

        # Apply position control to arm joints
        p.setJointMotorControlArray(
            bodyUniqueId=env.robot_id,
            jointIndices=env.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_q,
            forces=[100.0] * 4,
            physicsClientId=env.physics_client
        )

        # Apply angular gripper finger control
        left_angle = 0.4 * gripper_action
        right_angle = -0.4 * gripper_action
        p.setJointMotorControl2(env.robot_id, env.finger_indices[0], p.POSITION_CONTROL, targetPosition=left_angle, force=20.0, physicsClientId=env.physics_client)
        p.setJointMotorControl2(env.robot_id, env.finger_indices[1], p.POSITION_CONTROL, targetPosition=right_angle, force=20.0, physicsClientId=env.physics_client)

        # Step simulation and update env step
        _, _, _, _, _ = env.step(np.concatenate([np.zeros(4), [gripper_action]]))

        # Smooth 3D orbiting camera angle showing camera stand, angular gripper, object, and place cylinder
        angle = (step / total_frames) * np.pi * 0.5
        cam_x = 0.85 * np.cos(angle) + 0.1
        cam_y = 0.85 * np.sin(angle) + 0.1

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, 0.75],
            cameraTargetPosition=[0.0, 0.20, 0.20],
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

        _, _, rgb_img, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=env.physics_client
        )

        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        frames_bgr.append(bgr_array)
        pil_images.append(Image.fromarray(rgb_array))

    env.close()

    # 1. Encode MP4 Video using OpenCV
    os.makedirs(os.path.dirname(os.path.abspath(mp4_output)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_output, fourcc, fps, (width, height))

    for frame in frames_bgr:
        out.write(frame)
    out.release()
    print(f"[Video Recorder] MP4 video encoded and saved to '{mp4_output}' ({os.path.getsize(mp4_output)} bytes).")

    # 2. Encode Animated WebP using PIL
    os.makedirs(os.path.dirname(os.path.abspath(webp_output)), exist_ok=True)
    pil_images[0].save(
        webp_output,
        save_all=True,
        append_images=pil_images[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"[Video Recorder] Animated WebP saved to '{webp_output}' ({os.path.getsize(webp_output)} bytes).")

    # 3. Copy artifacts if directory provided
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        art_mp4 = os.path.join(artifact_dir, "sim_trajectory.mp4")
        art_webp = os.path.join(artifact_dir, "sim_trajectory.webp")

        out_art = cv2.VideoWriter(art_mp4, fourcc, fps, (width, height))
        for frame in frames_bgr:
            out_art.write(frame)
        out_art.release()

        pil_images[0].save(
            art_webp,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"[Video Recorder] Artifact video copies saved to '{artifact_dir}'.")

    return True


if __name__ == "__main__":
    art_dir = "/home/werrewulf/.gemini/antigravity-ide/brain/16bbf462-968d-4e6b-ab2b-1a7f9eb10d8f"
    record_simulation_video(
        mp4_output="data/sim_trajectory.mp4",
        webp_output="data/sim_trajectory.webp",
        artifact_dir=art_dir
    )
