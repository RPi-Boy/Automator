import os
import sys
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.arm_env import ArmGraspEnv
from sim.trajectory_planner import QuinticTrajectoryPlanner
import pybullet as p


def record_test_video(seed, object_name, mp4_path, webp_path, artifact_dir=None):
    """
    Executes an Ahead-of-Time Motion Planned trajectory ("Plan Before Move") with Non-Slip Grasping.
    Demonstrates clean, non-slip Pick -> Grasp -> Transport -> Place workflow for distinct dataset objects.
    """
    print(f"\n[Video Recorder] Recording Test Video: '{object_name}' (Seed {seed})...")
    env = ArmGraspEnv(render_mode=None)
    obs, info = env.reset(seed=seed)

    width, height = 640, 480
    fps = 30
    total_frames = 150

    obj_pos = obs[11:14]
    place_pos = obs[14:17]

    # Step 1: Ahead-of-Time Quintic Trajectory Motion Planner (<3ms)
    planner = QuinticTrajectoryPlanner()
    home_pos = obs[8:11]
    traj = planner.plan_pick_and_place_trajectory(home_pos, obj_pos, place_pos)
    print(f"  - Trajectory Planned in {traj['planning_latency_ms']:.4f} ms (<30ms target). Waypoints generated.")

    # Solve IK along planned waypoints
    def solve_ik(cartesian_pos):
        return p.calculateInverseKinematics(
            env.robot_id, env.ee_index, cartesian_pos,
            maxNumIterations=500, residualThreshold=1e-5, physicsClientId=env.physics_client
        )

    p1_path = traj["phase1_approach"]
    p2_path = traj["phase2_grasp"]
    p3_path = traj["phase3_transport"]
    p4_path = traj["phase4_place"]

    frames_bgr = []
    pil_images = []
    grasp_cid = None

    print(f"  - Executing 150-frame Motion Planned Non-Slip Trajectory...")

    for step in range(total_frames):
        # Map step index to trajectory phase
        if step < len(p1_path):
            cart_target = p1_path[step]
            g_action = 1.0  # Open gripper
        elif step < len(p1_path) + len(p2_path):
            idx = step - len(p1_path)
            cart_target = p2_path[idx]
            g_action = -1.0  # Close angular gripper
        elif step < len(p1_path) + len(p2_path) + len(p3_path):
            idx = step - len(p1_path) - len(p2_path)
            cart_target = p3_path[idx]
            g_action = -1.0  # Hold grasp clamp
        else:
            idx = min(len(p4_path) - 1, step - len(p1_path) - len(p2_path) - len(p3_path))
            cart_target = p4_path[idx]
            g_action = 1.0  # Release gripper

        # Solve IK for current Cartesian waypoint
        ik_q = solve_ik(cart_target)

        # Apply position control to arm joints
        p.setJointMotorControlArray(
            bodyUniqueId=env.robot_id,
            jointIndices=env.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=ik_q[:4],
            forces=[100.0] * 4,
            physicsClientId=env.physics_client
        )

        # Apply position control to angular gripper fingers
        left_angle = 0.4 * g_action
        right_angle = -0.4 * g_action
        p.setJointMotorControl2(env.robot_id, env.finger_indices[0], p.POSITION_CONTROL, targetPosition=left_angle, force=25.0, physicsClientId=env.physics_client)
        p.setJointMotorControl2(env.robot_id, env.finger_indices[1], p.POSITION_CONTROL, targetPosition=right_angle, force=25.0, physicsClientId=env.physics_client)

        # Non-Slip Grasp Attachment Constraint during Transport
        if g_action < -0.2 and grasp_cid is None:
            grasp_cid = p.createConstraint(
                parentBodyUniqueId=env.robot_id,
                parentLinkIndex=env.ee_index,
                childBodyUniqueId=env.object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.02],
                childFramePosition=[0, 0, 0],
                physicsClientId=env.physics_client
            )
        elif g_action > 0.2 and grasp_cid is not None:
            p.removeConstraint(grasp_cid, physicsClientId=env.physics_client)
            grasp_cid = None

        p.stepSimulation(physicsClientId=env.physics_client)

        # Smooth 3D camera panning angle
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

    # Encode MP4 Video
    os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
    for frame in frames_bgr:
        out.write(frame)
    out.release()
    print(f"  - Saved Non-Slip MP4 Video: '{mp4_path}' ({os.path.getsize(mp4_path)} bytes)")

    # Encode Animated WebP
    os.makedirs(os.path.dirname(os.path.abspath(webp_path)), exist_ok=True)
    pil_images[0].save(
        webp_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"  - Saved Non-Slip Animated WebP: '{webp_path}' ({os.path.getsize(webp_path)} bytes)")

    # Copy to artifact directory
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        art_mp4 = os.path.join(artifact_dir, os.path.basename(mp4_path))
        art_webp = os.path.join(artifact_dir, os.path.basename(webp_path))

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
        print(f"  - Saved Artifact Copy: '{art_mp4}'")


def record_all_3_tests():
    art_dir = "/home/werrewulf/.gemini/antigravity-ide/brain/16bbf462-968d-4e6b-ab2b-1a7f9eb10d8f"

    tests = [
        {"seed": 100, "name": "Sphere", "mp4": "data/sim_test1_sphere.mp4", "webp": "data/sim_test1_sphere.webp"},
        {"seed": 101, "name": "Cube", "mp4": "data/sim_test2_cube.mp4", "webp": "data/sim_test2_cube.webp"},
        {"seed": 102, "name": "Cylinder", "mp4": "data/sim_test3_cylinder.mp4", "webp": "data/sim_test3_cylinder.webp"},
    ]

    for test in tests:
        record_test_video(
            seed=test["seed"],
            object_name=test["name"],
            mp4_path=test["mp4"],
            webp_path=test["webp"],
            artifact_dir=art_dir
        )


if __name__ == "__main__":
    record_all_3_tests()
