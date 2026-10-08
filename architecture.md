# System Architecture

**Project:** Team Avatar Automator - shape-sorting 4-DOF arm, simulated in PyBullet, with perception and control running as ROS 2 nodes on a Jetson Orin Nano and a micro-ROS ESP32 as the actuator/safety controller.
**Status:** simulation is the product; the Jetson and ESP32 prove the edge pipeline. Items not yet run on hardware are marked *(unverified)*.

## 1. Data flow

```
 PyBullet scene (sim_bridge_node)                         Jetson Orin Nano
 +----------------------------+  /camera/image_raw  +---------------------------------+
 | floor, 4 bins, N objects,  | ------------------> | evidential_vision_node          |
 | 4-DOF arm + gripper        |  /camera/reference  |  1. background subtraction      |
 |                            | ------------------> |  2. crop each blob (128x128)    |
 |                            |                     |  3. MobileNetV3 (TensorRT/ONNX) |
 |                            |                     |     -> shape, xyz residual,     |
 |                            |                     |        Mahalanobis uncertainty  |
 |                            |                     +----------------+----------------+
 |                            |                                      | /arm_target [x,y,z,class,u]
 |                            |                     +----------------v----------------+
 |                            |   /joint_commands   | joint_trajectory_node           |
 |  SimArm (position control) | <------------------ |  analytic IK + quintic joint    |
 |                            |       /estop        |  trajectories, bin chosen by    |
 |                            | <------------------ |  class, u > 0.75 -> halt        |
 +----------------------------+                     +----------------+----------------+
                                                                      | /joint_commands, /arm_target,
                                                                      | /estop, /estop_reset (micro-ROS, serial)
                                                     +----------------v----------------+
                                                     | ESP32 (esp32_actuator.ino)      |
                                                     |  stepper trapezoids, gripper    |
                                                     |  servo, HW e-stop interrupt,    |
                                                     |  200 ms command watchdog        |
                                                     +---------------------------------+
```

The classical detector finds objects of *any* shape, so unknown objects are not silently ignored; the network only has to classify one centred crop and say whether it recognises it.

## 2. Perception

- **Detector** (`perception/detector.py`): difference against an empty-scene reference -> morphology -> connected components -> square crop + bounding-box features; blob centroid is back-projected to the table plane through `perception/camera.py` (numpy pinhole model shared by sim and Jetson).
- **Model** (`perception/mobilenet_uncertainty.py`): MobileNetV3-small features -> 128-d embedding. Heads: shape logits (sphere/cube/cylinder/brick) and a residual that refines the geometric xyz (it also sees the bbox, because a rescaled crop loses absolute size). ImageNet normalisation is inside the model.
- **Uncertainty**: minimum Mahalanobis distance to the class means with a tied covariance, fitted *after* training in eval mode. The raw distance is mapped to [0, 1] so that **0.75 corresponds to the 99th percentile of held-out in-distribution distances**; the whole computation is part of the exported graph (`DeployWrapper`), so TensorRT runs it.
- **Export**: `scripts/train_perception.py` writes `models/perception.pt` + `models/perception.onnx` (inputs `crop`, `bbox_feat`; outputs `logits`, `offset`, `uncertainty`; dynamic batch). Build the TensorRT engine **on the Jetson** with `scripts/build_trt_engine.sh`.

Complexity: Mahalanobis scoring is O(C * D^2) per crop (C = 4 classes, D = 128); fitting is O(N * D^2 + D^3).

## 3. Planning and control

- `kinematics.py`: closed-form IK for the 3-link planar chain plus yaw (shoulder at z = 0.25 m, links 0.30 m and 0.25 m, tool pitch pointing down, 0.09 m fingertips), clamped to the URDF joint limits. Verified against PyBullet forward kinematics to < 1 mm (`tests/test_kinematics.py`).
- **Workspace limit**: the elbow limit of +/-2.0 rad makes anything within ~0.30 m of the shoulder unreachable, so the pick region is r in [0.30, 0.38] m and bins sit at r = 0.33 m. Widening the elbow limit in the URDF removes this, if the real arm can fold further.
- `trajectory.py`: waypoints approach -> grasp -> lift -> transport -> release -> home, joint-space quintic segments whose durations respect 80% of the URDF joint velocity limits. The lift height is the lowest one that is IK-reachable above both pick and bin.
- The sequencer is plain numpy and is used both by `joint_trajectory_node` (50 Hz) and by `scripts/sort_demo.py`.
- Grasping in simulation is a fixed constraint created at the object's current pose when the gripper closes within 4.5 cm of the fingertips; the finger geometry itself does not hold objects.

## 4. Safety layers (all converge on one halt routine on the ESP32)

1. Hardware line on GPIO33 -> real interrupt.
2. Uncertainty > 0.75 on `/arm_target` (vision is paused while `/arm_busy`, because the moving arm itself is not in the background reference) (the vision node publishes the *most* uncertain object whenever one exceeds the threshold) or `/estop = true`.
3. 200 ms watchdog on `/joint_commands`.
4. `joint_trajectory_node` independently refuses to plan and publishes `/estop` for u > 0.75.

Latched until `/estop_reset` and the hardware line is released. Details and wiring: `firmware/esp32_microros/README.md`.

## 5. RL (SAC)

`sim/reach_env.py` trains SAC to put the fingertip within 1 cm of a random reachable target (14-d observation, 4-d joint-velocity action). It is a benchmark against the analytic IK; the sorting pipeline does **not** depend on it.

## 6. Layout

```
perception/   detector.py camera.py mobilenet_uncertainty.py inference.py
sim/          scene.py sim_arm.py reach_env.py arm_env.py(legacy) trajectory_planner.py(legacy) urdf/
scripts/      generate_sim_dataset.py train_perception.py sort_demo.py train_sac_reach.py
              benchmark_edge.py build_trt_engine.sh
ros2_ws/src/arm_perception/arm_perception/
              kinematics.py trajectory.py  joint_trajectory_node.py evidential_vision_node.py
              sim_bridge_node.py imx219_camera_node.py(real camera, not used in the sim demo)
firmware/esp32_microros/esp32_actuator/   Arduino sketch
tests/        kinematics, trajectory, perception, integration, sim
```
