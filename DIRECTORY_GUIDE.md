# Directory guide

What every folder and file in the repository is for, and where the code for each important part lives. "Tracked" means committed to git; "local only" means it exists on this PC but is git-ignored.

## 1. Where is the code for...

| I want to find... | File |
|---|---|
| The simulated world (objects, bins, camera, rendering) | `sim/scene.py` |
| The PyBullet arm (joint set-points, grasp weld) | `sim/sim_arm.py`, URDF in `sim/urdf/arm_4dof.urdf` |
| The end-to-end sorting loop (what the tests run) | `scripts/sort_demo.py` |
| Object detection (background subtraction, crops, table-plane xyz) | `perception/detector.py`, camera model in `perception/camera.py` |
| The neural network (MobileNetV3-small, xyz head, uncertainty score) | `perception/mobilenet_uncertainty.py` |
| Running the model (ONNX Runtime, PyTorch, TensorRT backends) | `perception/inference.py` |
| Picking which object to grasp next (worst uncertainty first) | `perception/targeting.py` |
| Inverse kinematics and forward kinematics | `ros2_ws/src/arm_perception/arm_perception/kinematics.py` |
| Pick-and-place motion (quintic joint trajectories) | `ros2_ws/src/arm_perception/arm_perception/trajectory.py` |
| Shared geometry numbers (bins, limits, tool length) | `ros2_ws/src/arm_perception/arm_perception/config.py` |
| ROS 2 nodes (vision, trajectory, simulator bridge, viewer) | `ros2_ws/src/arm_perception/arm_perception/*_node.py`, launch file in `ros2_ws/src/arm_perception/launch/sort_demo.launch.py` |
| ESP32 firmware (steppers, e-stop, watchdog, micro-ROS) | `firmware/esp32_microros/esp32_actuator/` |
| Training the perception model | `scripts/train_perception.py` (data from `scripts/generate_sim_dataset.py`) |
| Training the SAC reach policy | `scripts/train_sac_reach.py`, environment in `sim/reach_env.py` |
| Latency benchmarks | `scripts/benchmark_edge.py` |
| The web dashboard (server and page) | `webapp/server.py`, `webapp/static/index.html` |
| Unit and integration tests | `tests/` |
| Results and numbers | `report.md`, `docs/FULL_PROJECT_REPORT.md`, `docs/JETSON_RESULTS.md`, `models/*.json` |

## 2. Top level

| Path | What it is |
|---|---|
| `README.md` | Project summary, architecture figure, setup, the one reproduce command, team details (the BCSE306 front page). |
| `report.md` | Short results report. Measured numbers here are the reference for the other docs. |
| `architecture.md` | Data flow, perception, planning, safety layers, edge deployment, layout. |
| `instructions.md` | Step by step instructions for the juniors who run the project on the Jetson. |
| `helper.md` | How to run the simulation on this PC, seed commands, where videos go. |
| `DIRECTORY_GUIDE.md` | This file. |
| `Plan.md`, `task_plan.md`, `LEARNINGS.gemini.md` | Historical planning and notes from early in the project. Several details are superseded; the README and `docs/` are current. |
| `requirements.txt` | Python dependencies for the PC (training, simulation, ONNX). |
| `train_colab.ipynb` | Colab notebook used for GPU training. |
| `jetson_selfcheck.txt` | Output of the last self-check run (local only, written by `scripts/jetson_selfcheck.py`). |
| `client_secrets.json` | Local credential file (local only, git-ignored). Never commit it. |
| `.gitignore` | Ignores `.venv`, `data/`, `runs/`, checkpoints and the secrets above. Models in `models/` were added explicitly. |
| `.claude/` | Claude Code settings for this project (local permissions). |
| `.venv/` | Python virtual environment (local only, about 5.7 GB). |

## 3. `sim/` : the simulation (PyBullet)

| File | Purpose |
|---|---|
| `scene.py` | Shared scene: the four known shapes (sphere, cube, cylinder, brick), the held-out capsule used as the unseen object, one bin with a tray per class, the fixed camera, rendering. Used by the dataset generator, the demo and the ROS bridge. |
| `sim_arm.py` | The arm driven by `[q1..q4, gripper]` set-points, with a weld that holds the grasped object at the fingertips. |
| `reach_env.py` | Gymnasium environment for the SAC reach task (fingertip within 1 cm of a random target). |
| `arm_env.py`, `trajectory_planner.py` | Earlier environment and Cartesian planner. Kept for the older scripts and `tests/test_sim.py`; the current pipeline uses `scene.py`, `sim_arm.py` and `trajectory.py`. |
| `urdf/arm_4dof.urdf` | Robot description: 4 joints plus a gripper. |

## 4. `perception/` : vision and the model

| File | Purpose |
|---|---|
| `camera.py` | Numpy-only pinhole camera matching PyBullet's conventions. |
| `detector.py` | Fixed-camera detector: background subtraction, blobs, masked 128 x 128 crops, back-projection to table xyz. |
| `mobilenet_uncertainty.py` | The network: MobileNetV3-small backbone, shape classifier, xyz residual head, Mahalanobis out-of-distribution score. |
| `inference.py` | `load_backend(path)` returns an ONNX Runtime, PyTorch or TensorRT backend with one interface: `(crop, bbox_feat) -> (logits, offset, uncertainty)`. |
| `targeting.py` | Chooses the next target; shared by the ROS vision node and the simulator demo. |

## 5. `scripts/` : things you run

| Script | Use it to |
|---|---|
| `sort_demo.py` | Run the sorting simulation (trials, seeds, unseen objects, video, live frames, metrics). The engine behind every test. |
| `run_pc_sim.sh` | Run `sort_demo.py` on this PC with H.264 videos in `runs/pc_sim_videos/`. |
| `start_pc_dashboard.sh` | Start the web dashboard on this PC (section 8). |
| `start_dashboard.sh` | Start the web dashboard on the Jetson (uses system Python and ROS). |
| `benchmark_edge.py` | Per-stage latency (detect, network, IK and planning) for any backend. |
| `jetson_selfcheck.py` | Check that a machine can run the demo (`--skip-ros` on a PC). |
| `jetson_install.sh` | One-time dependency install on a fresh Jetson. |
| `build_trt_engine.sh` | Build a TensorRT engine from the ONNX model (must run on the Jetson). |
| `reproduce_jetson.sh` | The one-command Jetson reproduction (self-check, engines, benchmarks, trtexec, sorting run, summary). |
| `summarize_jetson_run.py` | Print and save the summary table for one `reproduce_jetson.sh` run. |
| `run_jetson_demo.sh` | Launch the ROS 2 simulated demo on the Jetson. |
| `start_microros_agent.sh` | Start the micro-ROS agent for the ESP32 on USB. |
| `generate_sim_dataset.py` | Render the training dataset to `data/sim_shapes.npz`. |
| `train_perception.py`, `train_sac_reach.py` | Train the perception model and the SAC policy. |
| `run_colab_training.py`, `train_rl_agent.py` | Older training entry points (Colab and the first RL setup). |
| `demo_gui.py`, `record_simulation_video.py`, `record_3_test_videos.py`, `render_simulation.py` | Older demo, rendering and recording scripts from before `sort_demo.py`. They produced the media in `data/`. |

## 6. `ros2_ws/` : ROS 2 Humble package

`ros2_ws/src/arm_perception/` is a normal ROS package (`package.xml`, `setup.py`, `setup.cfg`, `resource/`). The Python lives in `arm_perception/`:

| File | Purpose |
|---|---|
| `evidential_vision_node.py` | Camera frame to detections, class, refined xyz and uncertainty. Pauses while `/arm_busy` is true. |
| `joint_trajectory_node.py` | `/arm_target` to IK to quintic trajectory to `/joint_commands`; handles `/estop` and `/estop_reset`. |
| `sim_bridge_node.py` | Plays the physical world: runs PyBullet headless and talks ROS. |
| `sim_viewer_node.py` | Demo window: simulation view beside the perception view, keyboard e-stop. |
| `imx219_camera_node.py` | Real camera node for the IMX219 (not used in the sim-only demo). |
| `kinematics.py`, `trajectory.py`, `config.py`, `overlay.py`, `_paths.py` | Pure Python helpers with no ROS imports, so they are unit-tested anywhere. |

`launch/sort_demo.launch.py` starts the whole demo.

## 7. `firmware/` : ESP32

`firmware/esp32_microros/esp32_actuator/` is an Arduino IDE sketch: `esp32_actuator.ino` (micro-ROS node, watchdog, status heartbeat), `pins.h` (GPIO map, GPIO33 e-stop interrupt), `stepper_driver.h/.cpp` (AccelStepper wrapper). `firmware/esp32_microros/README.md` has the upload steps. It has not been compiled or flashed.

## 8. `webapp/` : the test dashboard

`server.py` is a standard-library HTTP server with a token and a fixed job whitelist (self-check, latency benchmark, live test, backend comparison, sorting run with video, trtexec, engine build, ROS snapshot). `static/index.html` is the page. Job output goes to `runs/<timestamp>_<job>/` (`log.txt`, `metrics.jsonl`, `meta.json`, `videos/`).

## 9. `models/` : trained artefacts

| File | What it is |
|---|---|
| `perception.onnx`, `perception.pt` | The perception model (ONNX is what the demo and the Jetson load). |
| `perception_metrics.json` | Accuracy, localisation error and OOD results. |
| `sac_reach/` | SAC checkpoints (`best_model.zip`, `final_model.zip`), `metrics.json`, logs. |
| `best_model.zip`, `best_sac_policy.zip`, `logs/` | Older SAC checkpoints and evaluation logs. |
| `sort_results.json`, `sort_unknown_results.json` | Reference 20-trial sorting results (known and unseen objects). |
| `sort_demo.mp4` | Reference sorting video. |
| `benchmark_dev_pc_onnx_cpu.json` | PC latency benchmark. |

TensorRT engines (`perception_fp16.engine`, `perception_fp32.engine`) are built on the Jetson and git-ignored.

## 10. `tests/`

`test_kinematics.py` (analytic FK against PyBullet, IK accuracy), `test_trajectory.py` (sequencer limits, halting, bin mapping), `test_overlay.py` (drawing helpers), `test_perception.py` (network forward pass, Mahalanobis fit), `test_sim.py` (the older reach/grasp environment), `test_integration.py` (uncertainty threshold, ONNX parity, detector geometry, end-to-end sorting of three objects). Run with `.venv/bin/python -m unittest discover -s tests` (pytest is not installed in the venv).

## 11. `docs/`

| File | Content |
|---|---|
| `FULL_PROJECT_REPORT.md` | Full report: data, training, testing, all results. |
| `JETSON_RESULTS.md` | Measured Jetson latency and sorting results, with raw files in `jetson_results/` (benchmark JSON, trtexec logs, one sorting video). |
| `PROBLEMS_AND_FIXES.md` | The 40 problems found and how each was fixed. |
| `Team34_BCSE306_FinalReview.pdf` | The submitted final review document. |

## 12. Data and outputs (local only)

| Path | What it is |
|---|---|
| `data/` | `sim_shapes.npz` (training set, 57 MB) and older demo media (`demo_cube`, `sim_test*`, `sim_trajectory`, `sim_render.png`). Git-ignored. |
| `runs/` | Everything generated by tests: dashboard runs (`<timestamp>_<job>/`), `pc_sim_videos/` from `run_pc_sim.sh`, `pc_check_onnx_20trials.*`, and while the PC dashboard runs in the background, `dashboard.pid` and `dashboard.log`. Git-ignored. |
| `notebooks/` | Empty. |

## 13. Running the tests from the website on this PC

```bash
scripts/start_pc_dashboard.sh            # foreground; Ctrl+C stops it
scripts/start_pc_dashboard.sh --bg       # background; stop with --stop
scripts/start_pc_dashboard.sh --local    # only reachable from this PC
scripts/start_pc_dashboard.sh --port 9000
```

It prints one URL per network interface, ending in `/?token=...`. Open one on this PC or on any device on the same network. The token is saved in `~/.automator_dashboard_token`, so the URL stays the same across restarts.

On a PC the page offers the ONNX Runtime (CPU) backend. Self-check, latency benchmark, live latency test and sorting run with video all work; the TensorRT jobs need a Jetson and are skipped. Anyone with the URL can run these tests on this PC, so use `--local` on a network you do not trust.
