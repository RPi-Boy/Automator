# Automator: shape-sorting arm (PyBullet + Jetson Orin Nano + ESP32)

A simulated 4-DOF arm sorts objects into bins **by shape**. The perception and control stack runs as ROS 2 nodes
on a Jetson Orin Nano (TensorRT), with an ESP32 (micro-ROS) as the actuator and safety controller.
Design and rationale: [architecture.md](architecture.md). Known problems and fixes: [docs/PROBLEMS_AND_FIXES.md](docs/PROBLEMS_AND_FIXES.md). Measured results: [report.md](report.md).

## Quick start (PC)
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/generate_sim_dataset.py            # render the dataset from the sim camera (~4 min)
.venv/bin/python scripts/train_perception.py --pretrained   # train, calibrate, evaluate, export ONNX (~10 min CPU)
.venv/bin/python scripts/sort_demo.py --trials 20 --video models/sort_demo.mp4
.venv/bin/python scripts/sort_demo.py --gui                 # watch it
.venv/bin/python -m unittest discover tests
```
Optional: `scripts/train_sac_reach.py` trains the SAC reach controller (1 cm success radius).

## Jetson + ESP32 (ROS 2 Humble)
`*.pt`, `*.onnx`, `*.engine` and `data/` are git-ignored, so the model does not arrive with a `git clone`.
Either copy `models/perception.onnx` to the Jetson (`scp models/perception.onnx jetson:~/Automator/models/`) or train there.
All commands below run from the repository root on the Jetson unless stated.
```bash
# one-time: Python 3.10 (JetPack). numpy/opencv come from apt (python3-numpy python3-opencv).
pip3 install pybullet onnxruntime              # onnxruntime-gpu from NVIDIA's Jetson wheels if you want its GPU provider
python3 -c "import tensorrt; print(tensorrt.__version__)"   # JetPack 6.x = TensorRT 10 (supported), 8.x also handled
# The TensorRT backend also needs NVIDIA's CUDA-enabled PyTorch wheel. Without it use the .onnx file (ONNX Runtime).

export AUTOMATOR_ROOT=$PWD                     # perception/ and sim/ live outside the ROS package
source /opt/ros/humble/setup.bash
(cd ros2_ws && colcon build --symlink-install --packages-select arm_perception) && source ros2_ws/install/setup.bash

scripts/build_trt_engine.sh fp16               # build the engine ON the Jetson -> models/perception_fp16.engine

# terminal 1: micro-ROS agent (ESP32 on /dev/ttyUSB0, see firmware/esp32_microros/README.md)
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
# terminals 2-4
ros2 run arm_perception sim_bridge_node --ros-args -p objects:=4
ros2 run arm_perception evidential_vision_node --ros-args -p model_path:=$AUTOMATOR_ROOT/models/perception_fp16.engine
ros2 run arm_perception joint_trajectory_node        # defaults: fingertip planning, 4 shape bins
# safety test: restart the bridge with one unseen mesh on the table
ros2 run arm_perception sim_bridge_node --ros-args -p objects:=4 -p inject_unknown:=true
# per-stage latency on the Jetson
python3 scripts/benchmark_edge.py --model models/perception_fp16.engine --tag jetson_fp16
```
Topics: `/camera/image_raw`, `/camera/reference`, `/arm_target`, `/arm_busy`, `/joint_commands`, `/estop`, `/estop_reset`.
The vision node pauses while `/arm_busy` is true (the moving arm would otherwise be scored as an unknown object), so an unknown
object that enters the workspace mid-motion is caught once the arm stops.
ESP32 sketch: `firmware/esp32_microros/esp32_actuator/` (Arduino IDE; wiring, pins and safety behaviour in its README).

## Layout
```
perception/   detector, camera model, MobileNetV3 + Mahalanobis model, inference backends (TensorRT/ONNX/torch)
sim/          sorting scene, simulated arm, URDF, SAC reach env (arm_env.py / trajectory_planner.py are legacy)
scripts/      dataset generation, training, sort demo, SAC training, edge benchmark, TensorRT build
ros2_ws/      arm_perception: kinematics, trajectory, joint_trajectory_node, evidential_vision_node, sim_bridge_node
firmware/     ESP32 micro-ROS sketch
tests/        kinematics (vs PyBullet FK), trajectory, perception, integration
```

## Status
Simulation, perception and planning are tested on the development PC. **Not yet run on hardware:** TensorRT engine,
ROS nodes under a live ROS 2, the ESP32 sketch (not compiled), the hardware e-stop wiring.
