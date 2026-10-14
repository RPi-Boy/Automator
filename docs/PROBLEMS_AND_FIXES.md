# Problems found in the original project, and what was done

Status: **Fixed** = changed and tested on the PC. **Written** = code exists but has not run on the target hardware. **Open** = still a problem.
(A tarball of the repository before these changes is in the Claude session scratchpad; the repository is now on GitHub (github.com/RPi-Boy/Automator).)

## A. Claims that the code did not support
| # | Problem | Fix | Status |
|---|---|---|---|
| 1 | `report.md` claimed clean pick-and-place and a 10.28 ms pipeline. The integration test actually printed **0.0% success, 0.363 m placement RMSE** and only passed because its assertion allowed 0.40 m. | Test replaced by calibration, ONNX-parity, detector-geometry and end-to-end sorting tests. `report.md` rewritten with measured numbers only. | Fixed |
| 2 | The "latency" figure added up three unrelated timings, including a PyBullet physics step, with random weights and a 224x224 dummy input. | `scripts/benchmark_edge.py` times detection, network and IK/planning on real frames, per backend; run it on the Jetson. | Fixed (measured on the Jetson, docs/JETSON_RESULTS.md) |
| 3 | The E-stop test used the class mean as its "in-distribution" sample (distance exactly 0) and compared an *unbounded* distance to 0.75. | Distance mapped to [0, 1] so 0.75 = 99th percentile of held-out in-distribution data; tested with proper held-out samples. | Fixed |

## B. Perception
| # | Problem | Fix | Status |
|---|---|---|---|
| 4 | Training data was random-noise images with a red square; the class label (`i % 5`) was unrelated to the image and xyz was a function of the square's pixel position. The model could not have learned anything useful. | New data from the sim camera (`generate_sim_dataset.py`): 4 shapes with random colour/size/pose/lighting, multi-object scenes, tray context, plus unseen objects (1000 procedural meshes in disjoint pools, and a held-out capsule) for OOD. | Fixed |
| 5 | `evidential_vision_node` used random weights, fitted Mahalanobis on `torch.randn` data, loaded no engine/checkpoint, fed BGR to an RGB model, skipped ImageNet normalisation, and logged at 30 Hz. | Rewritten: loads the trained model through TensorRT/ONNX/torch backends, handles rgb8/bgr8 and row stride, batches crops, rate-limited. | Fixed (ROS run: open) |
| 6 | Mahalanobis statistics were fitted on features collected *during* training (train-mode BatchNorm, changing weights). | Refit after training in eval mode on clean data; threshold calibrated on a separate validation split. | Fixed |
| 7 | `compute_mahalanobis_distance` looped in Python per sample and class and called `torch.tensor()` on a list of tensors (breaks autograd/export). | Vectorised einsum, part of the exported graph. | Fixed |
| 8 | `num_classes=5` with 3 object types; no pretrained weights; no detection (whole frame resized to 224x224). | 4 classes, ImageNet-pretrained MobileNetV3-small, background-subtraction detector + masked 128x128 crops. | Fixed |
| 9 | TensorRT export enabled FP16 by checking `platform_has_tf32`, wrote a **fake `.engine` file** when TensorRT was missing, and left two outputs without dynamic batch. | Export writes ONNX only (all inputs/outputs dynamic batch); `scripts/build_trt_engine.sh` builds the engine on the Jetson with `trtexec`. The fake engine file was deleted. | Fixed (FP16 and FP32 engines built and checked on the Jetson) |
| 10 | Crops included tray edges and neighbouring objects; with trays in view false e-stops rose from 0% to 7.5% in one image region. | Crops are masked to the object; dataset includes trays. | Fixed |
| 11 | Unseen shapes that look like a known class (a capsule next to the cylinder class) are not flagged. | Measured and reported. Unseen meshes are flagged essentially always. A center-loss variant collapsed and was dropped. | **Open** |

## C. Simulation and control
| # | Problem | Fix | Status |
|---|---|---|---|
| 12 | Objects spawned at r = 0.20-0.35 m, but with the URDF's +/-2.0 rad elbow limit anything within ~0.30 m of the shoulder is unreachable. | Pick region r = 0.30-0.38 m, bins at 0.33 m; IK feasibility checked on 2000 random cases. Widening the elbow limit remains an option if the real arm can fold further. | Fixed (by restricting) |
| 13 | The old SAC task (pick-and-place with `-distance` reward) never succeeded, `train_rl_agent.py` used `EvalCallback` without importing it (NameError), and the policy was never part of the pipeline. | New fingertip-reach task with a 1 cm success radius (`reach_env.py`, `train_sac_reach.py`); the sorting pipeline uses IK, not SAC. | Fixed (result in report.md) |
| 14 | The grasp weld teleported the object to a fixed offset from the wrist, and objects rolled forever after release (no rolling friction). | Weld at the object's current relative pose; rolling friction/damping; shallow rimmed trays. | Fixed |
| 15 | The old planner produced Cartesian waypoints with no IK, so nothing could drive joints. | Closed-form IK (verified against PyBullet FK < 1 mm) + joint-space quintic segments limited to 80% of URDF velocity limits. | Fixed |
| 16 | The camera was a decorative marker 0.8 m from the objects (about 17 px per object). | Real fixed camera, 640x480, ~1.3 px/mm, shared by dataset, demo and the Jetson node via `perception/camera.py`. | Fixed |

| 16b | Objects whose silhouettes overlap in the oblique camera view merge into one blob and are always flagged as unknown (10 of 10 merged blobs). | Demo spawns objects apart in image space; documented. In the real system a merged blob stops the arm (conservative). | Fixed in sim / by design |
| 16c | Running vision while the arm moves flagged the arm's own links as unknown objects (u > 0.75 in 1 of 3 sampled runs). | `/arm_busy` from `joint_trajectory_node`; vision pauses while busy plus a 0.6 s settle. Tradeoff documented. | Written |

## D. ROS 2 / Jetson
| # | Problem | Fix | Status |
|---|---|---|---|
| 17 | `ros2 run arm_perception ...` would fail after `colcon build`: no `resource/arm_perception` marker, no `setup.cfg` script dirs. | Added both. | Written |
| 18 | The vision node's `sys.path` hack pointed at the wrong place from an installed package. | `AUTOMATOR_ROOT` env var or upward search (works with `--symlink-install`). | Written |
| 19 | The only image source was the real IMX219; nothing showed sim objects. | `sim_bridge_node` renders the sim scene and applies `/joint_commands`. | Written |
| 19b | A review found the ROS node's defaults did not match what the demo measured: it planned for the wrist (fingers 9 cm through the floor) and its empty bin parameter sent every class to one bin. | One shared `arm_perception/config.py`; the node, demo and tests all build `PickPlaceSequencer()` with no arguments; a test checks fingertip-at-object and class-to-bin. | Fixed |
| 20 | Dev venv is Python 3.14; Jetson/Humble is Python 3.10. | New node code avoids 3.11+ syntax; compile-checked only on 3.14. | Partly verified (benchmarks and sorting ran on the Jetson's Python 3.10.12; ROS nodes not yet run) |

## E. ESP32 firmware
| # | Problem | Fix | Status |
|---|---|---|---|
| 21 | No micro-ROS at all: `loop()` fed itself a hardcoded fake target. | Real subscriptions to `/joint_commands`, `/arm_target`, `/estop`, `/estop_reset`, preallocated message memory. | Written |
| 22 | "Hardware ISR" was an ordinary function call; no interrupt was attached. | `attachInterrupt` on GPIO33; the software paths call the same halt routine; direct register writes only. | Written |
| 23 | `Serial.printf` everywhere, while micro-ROS owns `Serial` as its transport. | All prints removed. | Written |
| 24 | `MOTOR_ENABLE_PIN = 12` is the flash-voltage strapping pin (can prevent boot). | Moved to 25; pin map avoids all boot/flash pins. | Written |
| 25 | No motor control and no way to clear an e-stop; no watchdog. | AccelStepper trapezoids on core 1, micro-ROS on core 0, 200 ms watchdog, latched e-stop with a reset topic. | Written |
| 26 | `espressif/esp32-camera` in `lib_deps` was unused. | Removed; PlatformIO config dropped in favour of the Arduino IDE sketch you use. | Fixed |
| 27 | Gear ratios/steps-per-radian are unknown. | Placeholders, flagged in the code and README. **Calibrate before powering the arm.** | **Open** |
| 28 | Disabling drivers on any stop removes holding torque. | Documented; needs brakes or a safe pose on the real arm. | **Open** |

## F. Repository hygiene
| # | Problem | Fix | Status |
|---|---|---|---|
| 29 | `client_secrets.json` (Google OAuth) sat untracked and not ignored; no commit existed. | Added to `.gitignore` (also `token.json`, `.pio/`). **Rotate that secret if it was ever shared.** | Fixed |
| 30 | Docs disagreed with the code (14-d vs 18-d state, a `notebooks/` directory that does not exist, `data/` links, "<1 ms ISR" claims). | `architecture.md`, `README.md`, `task_plan.md` rewritten; `report.md` contains only measured values. | Fixed |
| 31 | Dead code and stale artefacts: Cornell-noise dataset code/tests, a placeholder engine, an old ONNX file. | Removed (the pre-change tarball has them). `train_colab.ipynb`, `record_*` scripts and `arm_env.py` still target the old pick-and-place task. | **Open** (stale, unused) |

## G. Found while bringing the Jetson up (2026-10-13)
| # | Problem | Fix | Status |
|---|---|---|---|
| 32 | `pip install onnxruntime` pulls **NumPy 2**, which breaks the apt OpenCV (`_ARRAY_API not found`). | Pin `numpy<2` in `jetson_install.sh`, instructions and troubleshooting. | Fixed |
| 33 | A fresh JetPack 6 image had **no TensorRT or CUDA**, and after `apt install tensorrt` it still failed (`libnvdla_compiler.so`, then `libcudla.so.1`: cannot open shared object file). | `jetson_install.sh` installs `tensorrt python3-libnvinfer libnvinfer-bin nvidia-l4t-dla-compiler libcudla-12-6` and runs `ldconfig`. | Fixed |
| 34 | The TensorRT backend needed NVIDIA's CUDA PyTorch wheel, which is not available on the board; it was also written for TensorRT 8 binding calls. | Rewritten for the TensorRT 10 tensor API using `cuda-python`; verified against ONNX outputs (logit diff < 0.005, same classes). | Fixed |
| 35 | `ffmpeg` missing, and OpenCV writes MPEG-4 part 2, which browsers cannot play. | `ffmpeg` in the install script; the dashboard transcodes every video to H.264 (about 10x smaller). | Fixed |
| 36 | `sort_demo.py` imported PyTorch at the top, so it could not run on a Jetson without it. | It now uses the shared inference backends (.onnx / .engine / .pt). | Fixed |
| 37 | The instructions said to benchmark in "full-speed mode" but never said what was expected; numbers depend strongly on the power mode (CPU path 14.0 ms in 25 W vs 11.0 ms in MAXN_SUPER). | Documented; the measured table is in `docs/JETSON_RESULTS.md`. | Fixed |
| 38 | The simulation is very slow on the Jetson CPU (camera render 270-350 ms/frame; a 3-trial sorting run took 711 s), so the simulated camera cannot keep a 3 Hz rate. | Documented; use `camera_hz` 2-3 and expect slower-than-real-time simulation. Perception itself is about 5.6 ms. | Open (inherent) |
| 39 | This PC could not reach the Jetson: its SSH alias pointed at 172.17.221.225, which the PC routes into the `docker0` bridge (172.17.0.0/16). | Use the Jetson's real address (10.204.202.225) or change the Docker bridge subnet. | Noted |
| 40 | Juniors could not run the on-device tests (no access to the Jetson). | Web dashboard (`webapp/`) so one person can run and download everything from a browser; the tests were then run on the Jetson and recorded. | Fixed |
