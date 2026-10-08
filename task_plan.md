# Task Plan: End-to-End Perception & RL Grasping Pipeline

## Phase 1: Workspace & Digital Twin Scaffolding
- [x] Create implementation plan & obtain user approval
- [x] Initialize `task_plan.md`, `README.md`, `LEARNINGS.gemini.md`, `requirements.txt`, and `.gitignore`
- [x] Implement 4-DOF manipulator URDF (`sim/urdf/arm_4dof.urdf`)
- [x] Implement PyBullet Gymnasium environment (`sim/arm_env.py`)
- [x] Implement SAC RL policy training script (`scripts/train_rl_agent.py`)
- [x] Scaffold Google Colab GPU notebook (`notebooks/train_colab.ipynb`) & script (`scripts/run_colab_training.py`)
- [x] Configure Colab CLI & upload notebook to Google Colab GPU platform (`train_colab.ipynb`)
- [x] Sync Google Colab GPU training outputs back to local workspace & perform performance analysis

## Phase 2: Few-Shot Perception & Uncertainty Engine
- [x] Implement MobileNetV3 + Mahalanobis distance model (`perception/mobilenet_uncertainty.py`)
- [x] Implement dataset downloader & synthetic fallback generator (`scripts/download_dataset.py`)
- [x] Implement PyTorch dataset & dataloader for Cornell Grasping Dataset (`perception/dataset.py`)
- [x] Implement dataset unit test suite (`tests/test_dataset.py`)
- [x] Implement ONNX / TensorRT FP16 quantization & export pipeline (`scripts/train_and_export_trt.py`)

## Phase 3: Hardware Integration (ROS2 & micro-ROS)
- [x] Scaffold ROS2 Humble package (`ros2_ws/src/arm_perception/`)
- [x] Implement ROS2 `imx219_camera_node.py`
- [x] Implement ROS2 `evidential_vision_node.py`
- [x] Implement ESP32 micro-ROS firmware stub with E-Stop ISR trigger (`firmware/esp32_microros/`)

## Phase 4: Verification & Documentation
- [x] Implement unit test suite (`tests/test_sim.py`, `tests/test_perception.py`)
- [x] Execute static analysis & compilation checks across all Python modules
- [x] Create system architecture specification (`architecture.md`)
- [x] Update `LEARNINGS.gemini.md` and generate `walkthrough.md`

## Phase 5: Hardware-in-the-Loop (HIL) & Real-Time Trajectory Controller
- [x] ROS2 `joint_trajectory_node.py` (target -> analytic IK -> quintic joint trajectory -> `/joint_commands` @ 50 Hz, `/estop`) - IK verified vs PyBullet FK (< 1 mm), sequencer unit-tested; node itself not run under ROS yet
- [x] Multi-axis stepper driver (`firmware/esp32_microros/esp32_actuator/stepper_driver.{h,cpp}`) - *unverified on hardware; steps/rad are placeholders*
- [x] ESP32 sketch: real micro-ROS subscriptions, GPIO e-stop interrupt, 200 ms watchdog, reset topic - *not compiled yet (Arduino IDE)*
- [x] ROS packaging fixed (`resource/` marker, `setup.cfg`, import path) - *needs a `colcon build` on the Jetson*

## Phase 6: System Integration, End-to-End E-Stop Verification & Benchmarking
- [x] Implement PyBullet offscreen simulation renderer & snapshot generator (`scripts/render_simulation.py`)
- [x] Implement direct position control contact video recording (`scripts/record_simulation_video.py`)
- [x] Implement 3D camera position & orientation frustum visual indicator (`sim/arm_env.py`, `scripts/render_simulation.py`)
- [x] Implement end-to-end integration test suite (`tests/test_integration.py`)
- [x] Implement system latency & throughput benchmarking script (`scripts/benchmark_pipeline.py`)
- [x] Implement HER goal-conditioned SAC policy optimization upgrade proposal
- [x] Verify execution, update `LEARNINGS.gemini.md`, and generate final `walkthrough.md`

## Phase 7: 2-Finger Gripper Mechanism & Pick-and-Place Manipulation
- [x] Upgrade URDF with active 2-finger angular gripper (`sim/urdf/arm_4dof.urdf`)
- [x] Upgrade Gymnasium environment to 5D action space with dynamic multi-object pick-and-place (`sim/arm_env.py`)
- [x] Implement high-precision closed-loop trajectory controller (<1.0mm accuracy)
- [x] Generate 3 separate video recordings for 3 distinct object tests: Sphere, Cube, and Cylinder (`scripts/record_3_test_videos.py`)
- [x] Run automated test suite & verify multi-object pick-and-place task execution

## Phase 8: Trajectory Motion Planning, Non-Slip Grasping & Latency Benchmarks
- [x] Implement Quintic Trajectory Motion Planner (`sim/trajectory_planner.py`)
- [x] Enforce non-slip contact dynamics & grasp constraint attachment in `sim/arm_env.py`
- [x] Verify total pipeline latency is <= 30ms max (`scripts/benchmark_pipeline.py`)
- [x] Update SAC RL retraining notebook for 5D pick-and-place policy (`notebooks/train_colab.ipynb`)
- [x] Run 500k timesteps SAC training with EvalCallback and anti-overfitting safeguards (`scripts/run_colab_training.py`)
- [x] Re-record all 3 test videos with planned smooth trajectory & non-slip object pick-and-place (`scripts/record_3_test_videos.py`)
- [x] Generate comprehensive master project report (`report.md`)


## Phase 9: Shape Sorting, Diverse Perception Data, Strict Tolerances
- [x] Sim-camera dataset generator with randomised colour/size/pose/lighting and held-out OOD objects (`scripts/generate_sim_dataset.py`)
- [x] Background-subtraction detector + numpy camera model usable on the Jetson (`perception/detector.py`, `camera.py`)
- [x] Retrained MobileNetV3 (4 shapes, xyz refiner, calibrated Mahalanobis) with ONNX export (`scripts/train_perception.py`)
- [x] End-to-end sort-by-shape demo with one bin per shape (`scripts/sort_demo.py`)
- [x] `sim_bridge_node.py` so the Jetson pipeline runs against the sim over ROS topics
- [x] SAC reach policy trained (`sim/reach_env.py`, `scripts/train_sac_reach.py`) - **1 cm target not met: 16% success after 300k steps**
- [ ] Build the TensorRT engine and run `scripts/benchmark_edge.py` on the Jetson
- [ ] Compile/flash the ESP32 sketch, calibrate `steps_per_rad`, wire the hardware e-stop
- [ ] Capsule-class OOD detection (see report.md)
