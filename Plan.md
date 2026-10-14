> **Original plan (historical).** The project has moved on: it is now a shape-sorting simulation with an edge pipeline on a Jetson. See `README.md`, `architecture.md` and `task_plan.md` for the current state.

# Project: End-to-End Perception & RL Grasping Pipeline
**Developer:** Siddhant Kumar (Sidd)
**Hardware:** Jetson Orin Nano, ESP32, 4-DOF Robotic Arm, IMX219 Camera

## 1. Workspace & Tooling Initialization
*Goal: Set up the local agentic environment and cloud bridge.*
- [ ] **Initialize Antigravity IDE:** Create the local workspace (`~/team-avatar-arm`). 
- [ ] **Spawn Agents:** Open the Antigravity **Manager View**. Spawn two parallel agents: one focused on `Robotics/ROS2` and one focused on `Deep Learning/RL`.
- [ ] **Configure Colab CLI:** 
  - Authenticate the CLI using the remote copy-paste flow (`colab auth`).
  - Test remote provisioning from the Antigravity terminal: `colab --gpu T4` to ensure seamless cloud-GPU bridging.

## 2. Phase 1: Digital Twin & Sim-to-Real RL
*Goal: Train the 4-DOF arm to calculate dynamic trajectories using Reinforcement Learning.*
- [ ] **Sim Environment (Local Vibe Coding):** 
  - **Prompt:** *"Scaffold a PyBullet/Isaac Sim environment using our 4-DOF arm URDF. Define the action space (joint torques) and observation space (current joint angles, target xyz coordinates)."*
  - **Action:** Review the generated simulation code Artifacts in the Antigravity Editor View.
- [ ] **RL Policy Scripting:** 
  - **Prompt:** *"Write a Soft Actor-Critic (SAC) training loop using Stable Baselines3. Reward function should penalize jerky movements and reward distance-to-target."*
- [ ] **Remote Execution:** Offload the heavy RL simulation to Colab without leaving the Antigravity terminal:
  ```bash
  colab --gpu A100 exec python scripts/train_rl_agent.py

    [ ] Artifact Recovery: Pull the trained policy weights back to your local Jetson workspace:
    Bash

    colab download best_sac_policy.zip

3. Phase 2: Few-Shot Perception & Uncertainty

Goal: Train the MobileNetV3 feature extractor and compile it for low-latency edge inference.

    [ ] Model Architecture (Local Vibe Coding):

        Prompt: "Write a PyTorch pipeline for a MobileNetV3 backbone. Implement Mahalanobis distance calculation to output both few-shot classification and out-of-distribution uncertainty."

    [ ] Remote Training & Optimization: Train the model on the Cornell Grasping Dataset and compile it using NVIDIA TensorRT for the Jetson.
    Bash

    # Spin up a T4 and run the training/export pipeline
    colab --gpu T4 exec python scripts/train_and_export_trt.py --quantize FP16

    [ ] Troubleshooting: If the TensorRT export fails, pull the logs directly into Antigravity for the agent to fix:
    Bash

    colab log > trt_error.log

        Prompt Agent: "Analyze trt_error.log and fix the ONNX-to-TensorRT conversion script."

4. Phase 3: Hardware Integration (ROS2 & micro-ROS)

Goal: Bridge the AI models to the physical hardware actuators.

    [ ] ROS2 Workspace (Async Agent Task):

        While Colab is training the RL model in the background, use the Manager View to instruct the Robotics agent: "Create a ROS2 Humble workspace. Write an imx219_camera_node and an evidential_vision_node that runs the .trt engine and publishes [x, y, z, class, uncertainty] to /arm_target."

    [ ] ESP32 Firmware (micro-ROS):

        Prompt: "Write micro-ROS C++ firmware for an ESP32. Subscribe to /arm_target. If uncertainty > 0.75, trigger an ISR to halt stepper motors. Else, pass the coordinates to the SAC RL policy for joint actuation."

5. Vibe Coding Best Practices for Project Avatar

    Trust & Verify via Artifacts: Instead of reading every line of tool-call code, have the Antigravity agents generate markdown "Implementation Plans" and "Architecture Diagrams" as Artifacts. Verify the logic, then approve the code generation.

    Asynchronous Parallelism: Use the Colab CLI's remote execution to your advantage. Start a 4-hour RL training job using colab exec, and immediately shift to the Antigravity Editor to vibe code the ESP32 hardware interrupts while the cloud GPU does the heavy lifting.

    Debugging with colab repl: If an agent-written script behaves weirdly on the remote GPU, drop into the active session interactively using colab repl to inspect the tensors manually.