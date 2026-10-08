import os
import sys
import time
import argparse
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.arm_env import ArmGraspEnv
from sim.trajectory_planner import QuinticTrajectoryPlanner
import pybullet as p


def run_interactive_demo(object_type="cube", gui=True, save_video=True):
    """
    Runs a live interactive PyBullet demo of the 4-DOF manipulator with Angular 2-Finger Gripper
    executing a motion-planned pick-and-place operation, saving the video output.
    """
    render_mode = "human" if gui else None
    print(f"\n==================================================")
    print(f" 🤖 4-DOF MANIPULATOR DEMO: Object = '{object_type.upper()}' | GUI = {gui}")
    print(f"==================================================")

    # Seed mapping: 100 -> Sphere, 101 -> Cube, 102 -> Cylinder
    seed_map = {"sphere": 100, "cube": 101, "cylinder": 102}
    seed = seed_map.get(object_type.lower(), 101)

    env = ArmGraspEnv(render_mode=render_mode)
    obs, info = env.reset(seed=seed)

    width, height = 640, 480
    fps = 30
    total_frames = 150

    obj_pos = obs[11:14]
    place_pos = obs[14:17]

    # Ahead-of-Time Trajectory Planner
    planner = QuinticTrajectoryPlanner()
    home_pos = obs[8:11]
    traj = planner.plan_pick_and_place_trajectory(home_pos, obj_pos, place_pos)
    print(f"[Motion Planner] Trajectory planned in {traj['planning_latency_ms']:.4f} ms (<30ms limit target).")

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

    print("[Demo Execution] Running Pick -> Grasp -> Transport -> Place trajectory...")

    for step in range(total_frames):
        if step < len(p1_path):
            cart_target = p1_path[step]
            g_action = 1.0
        elif step < len(p1_path) + len(p2_path):
            idx = step - len(p1_path)
            cart_target = p2_path[idx]
            g_action = -1.0
        elif step < len(p1_path) + len(p2_path) + len(p3_path):
            idx = step - len(p1_path) - len(p2_path)
            cart_target = p3_path[idx]
            g_action = -1.0
        else:
            idx = min(len(p4_path) - 1, step - len(p1_path) - len(p2_path) - len(p3_path))
            cart_target = p4_path[idx]
            g_action = 1.0

        ik_q = solve_ik(cart_target)

        p.setJointMotorControlArray(
            bodyUniqueId=env.robot_id,
            jointIndices=env.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=ik_q[:4],
            forces=[100.0] * 4,
            physicsClientId=env.physics_client
        )

        left_angle = 0.4 * g_action
        right_angle = -0.4 * g_action
        p.setJointMotorControl2(env.robot_id, env.finger_indices[0], p.POSITION_CONTROL, targetPosition=left_angle, force=25.0, physicsClientId=env.physics_client)
        p.setJointMotorControl2(env.robot_id, env.finger_indices[1], p.POSITION_CONTROL, targetPosition=right_angle, force=25.0, physicsClientId=env.physics_client)

        if g_action < -0.2 and grasp_cid is None:
            grasp_cid = p.createConstraint(
                parentBodyUniqueId=env.robot_id, parentLinkIndex=env.ee_index,
                childBodyUniqueId=env.object_id, childLinkIndex=-1,
                jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.02], childFramePosition=[0, 0, 0],
                physicsClientId=env.physics_client
            )
        elif g_action > 0.2 and grasp_cid is not None:
            p.removeConstraint(grasp_cid, physicsClientId=env.physics_client)
            grasp_cid = None

        p.stepSimulation(physicsClientId=env.physics_client)
        if gui:
            time.sleep(1.0 / 60.0)

        # Capture camera frame for video output
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
            fov=60.0, aspect=1.33, nearVal=0.1, farVal=3.0, physicsClientId=env.physics_client
        )

        _, _, rgb_img, _, _ = p.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER, physicsClientId=env.physics_client
        )

        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        frames_bgr.append(bgr_array)
        pil_images.append(Image.fromarray(rgb_array))

    env.close()

    if save_video:
        mp4_path = f"data/demo_{object_type}.mp4"
        webp_path = f"data/demo_{object_type}.webp"

        os.makedirs("data", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        for frame in frames_bgr:
            out.write(frame)
        out.release()

        pil_images[0].save(webp_path, save_all=True, append_images=pil_images[1:], duration=int(1000 / fps), loop=0)

        print("\n--------------------------------------------------")
        print(" 📹 DEMO VIDEO OUTPUTS GENERATED SUCCESSFULLY")
        print("--------------------------------------------------")
        print(f"  - MP4 Video File:  '{mp4_path}' ({os.path.getsize(mp4_path)} bytes)")
        print(f"  - WebP Animation:  '{webp_path}' ({os.path.getsize(webp_path)} bytes)")
        print("==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-DOF Robotic Arm Live Pick-and-Place Demo")
    parser.add_argument("--object", type=str, default="cube", choices=["sphere", "cube", "cylinder"], help="Dataset object type to manipulate")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without opening GUI window")
    args = parser.parse_args()

    run_interactive_demo(object_type=args.object, gui=not args.no_gui, save_video=True)
