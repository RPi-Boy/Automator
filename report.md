# Results report

Every number below was measured by the scripts in this repository, on one of two machines: the development PC (Arch Linux, Intel i7-1360P, 16 threads, CPU only,
Python 3.14) or the **Jetson Orin Nano Super** (JetPack 6 / L4T R36.5, TensorRT 10.3, Python 3.10), as labelled. PC results are stored in `models/*.json`; Jetson results are in
`docs/jetson_results/` and written up in `docs/JETSON_RESULTS.md`. **The ROS 2 nodes and the ESP32 have not been run** (the arm is not built, and only the simulation was tested on the
Jetson). Earlier versions of this report quoted results (10.28 ms pipeline latency, clean pick-and-place) that the code did not support; see `docs/PROBLEMS_AND_FIXES.md`.

## 1. Task
A simulated 4-DOF arm picks objects from a table and places each in the bin for its shape (sphere, cube, cylinder, brick). A fixed camera
sees the table; perception returns shape, position and an out-of-distribution (OOD) score; an OOD score above 0.75 stops the arm.

## 2. Perception (`scripts/train_perception.py`, `models/perception_metrics.json`)
Data (`scripts/generate_sim_dataset.py`): 16,464 object crops from 7,000 rendered scenes - random colour, size, yaw, position, 8 light directions, 1-4
objects, trays in view. Splits are by scene: train 10,289 / val 2,134 / test 2,217 crops. OOD: 918 (val) and 906 (test) crops of objects never used in
training - procedural meshes from two disjoint pools of `pybullet_data/random_urdfs`, and a capsule. Model: ImageNet-pretrained MobileNetV3-small, 20 epochs.

| Metric (held-out test) | Strict target | Measured | Met |
|---|---|---|---|
| Shape accuracy (2,217 crops) | >= 99% | **99.91%** (2 errors: cube->brick, cylinder->brick) | yes |
| xyz error, mean | <= 5 mm | **1.97 mm** (detector geometry alone: 5.43 mm) | yes |
| xyz error, 95th percentile | | 4.17 mm | |
| OOD AUROC, all unseen objects (906) | >= 0.95 | **0.934** | **no** |
| OOD AUROC, unseen procedural meshes | | 0.999 (95.4% flagged at the 0.75 threshold) | |
| OOD AUROC, capsule (held-out shape) | | **0.714** (13.6% flagged) | |
| OOD false-positive rate at 95% in-distribution acceptance | <= 10% | **15.6%** | **no** |
| False e-stop rate on known shapes at threshold 0.75 | | 0.72% | |

Reading: arbitrary unseen objects are caught reliably; an unseen shape that looks like a known class (a capsule next to the cylinder class) mostly is not.
One attempt to tighten the embedding (a center-loss term) made the features collapse and was dropped. The ">= 0.95" target is therefore not met on the
combined OOD set, and the capsule weakness is an open item.
The 0.75 threshold is calibrated: it is the 99th percentile of in-distribution Mahalanobis distances on the validation split.

## 3. End-to-end sorting in simulation (`scripts/sort_demo.py`)
The demo replays the ROS dataflow in one process: control at 60 Hz with `PickPlaceSequencer()` built with the node's defaults (fingertip planning, 4 shape bins);
vision at 5 Hz **only while the arm is idle and settled** (as `evidential_vision_node` does), worst-OOD-first target choice, re-detection after every pick.
Grasping uses a fixed constraint at the fingertips (the finger geometry does not hold objects). Success = a known object ends inside its own bin's tray.
Objects are spawned at least 11 cm apart *and* 100 px apart in the image, because touching silhouettes merge into one blob that the OOD score (correctly) calls
unfamiliar: of 590 detected blobs in 4-object scenes without that constraint, 10 held two objects and all 10 were flagged.

| Test | Result |
|---|---|
| 20 trials x 4 known objects (seeds 1000-1019) | **74 / 80 = 92.5%** in the correct bin, 0 trials ended in an e-stop |
| ... the 6 failures | all 6 objects ended outside their own tray (one brick ended in the sphere bin, i.e. a real misclassification); causes of the others were not isolated |
| 30 trials, 3 known + 1 unseen random mesh (seeds 5000-5029) | e-stop in **25 / 30** trials, all 25 on the unseen object and none on a known one; in the other 5 trials the unseen mesh was in the pick area but the detector never saw it; the arm moved an unseen object in 1 / 30 trials |
| Same sorting run **on the Jetson**, TensorRT FP16 (3 trials x 4 objects, seeds 1000-1002) | **12 / 12** in the correct bin, 0 e-stops; 3 H.264 videos (about 650 kB each); the run took 711 s because the simulator is CPU-bound on the Jetson (perception was about 5.6 ms of each cycle) |

`models/sort_demo.mp4` shows trial 0 from a spectator camera.
Tradeoff: because perception pauses while the arm moves, an unknown object that appears mid-motion is only caught once the arm stops. (With vision running during motion,
1 of 3 sampled runs scored u > 0.75 from the arm's own links crossing the pick area, so the pause is necessary with a background-subtraction detector.)

## 4. Latency (`scripts/benchmark_edge.py`, 3 objects per frame)
Stages: background-subtraction detection, network on a batch of crops (includes host-to-device copies and Python overhead), IK + trajectory planning. Simulator rendering is excluded.

### 4.1 Jetson Orin Nano Super (measured on the device)
500 iterations per backend, run through the web dashboard, power mode MAXN_SUPER with `jetson_clocks`:

| Backend | Detect | Network | Plan | **Total mean** | p95 | max |
|---|---|---|---|---|---|---|
| ONNX Runtime, CPU | 4.14 ms | 5.94 | 1.17 | **11.25 ms** | 13.84 | 27.61 |
| TensorRT FP32 (GPU) | 3.04 | 2.04 | 1.08 | **6.16 ms** | 6.48 | 6.80 |
| **TensorRT FP16 (GPU)** | 3.00 | 1.84 | 1.07 | **5.91 ms** | 6.20 | 6.83 |

- TensorRT cuts the network step 3.2x (5.94 to 1.84 ms) and the whole path 1.9x, and removes the CPU jitter (worst case 27.6 ms down to 6.8 ms). FP16 is only slightly faster than FP32 for a network this small.
- FP16 outputs match the FP32 CPU model to within 0.005 in the logits, with identical predicted classes (batch 1, 3 and 8).
- **Pure GPU time** (`trtexec`, batch 3, 300 iterations): FP16 **0.892 ms** (1117 qps), FP32 1.096 ms (910 qps); batch 1: 0.780 / 0.919 ms. The pipeline's 1.84 ms for the network step is about twice the GPU time because of host-device copies and Python.
- **Power mode matters:** in the original 25 W mode the CPU path took **14.02 ms** (p95 14.93; detect 4.75, network 7.71, plan 1.56) against 11.0 ms in MAXN_SUPER (200-iteration runs).
- **Sustained test** (60 s at 30 Hz, about 1,800 frames each):

| Backend | Total mean | p95 | max | first vs second half | GPU avg | CPU avg | Power avg (max) | Temp |
|---|---|---|---|---|---|---|---|---|
| TensorRT FP16 | 6.03 ms | 6.31 | 6.72 | 6.01 to 6.04 | 2 % | 7 % | **7.0 W** (7.7) | 48 C |
| ONNX Runtime CPU | 10.17 ms | 10.63 | 17.84 | 10.18 to 10.15 | 0 % | 44 % | 8.5 W (8.7) | 47 to 49 C |

  No drift or thermal throttling in 60 s (too short to find thermal limits). TensorRT leaves the CPU almost idle and draws less power while being faster.
- Engine build on the Jetson: FP16 195 s, FP32 85 s. The ONNX Runtime wheel for aarch64 is CPU-only, so the GPU path is the native TensorRT backend (`cuda-python`, no PyTorch).
- The simulated camera render alone takes 270-350 ms per frame on the Jetson CPU, so the simulator is far slower than real time there; this is a simulation cost, not a perception cost.

### 4.2 Development PC (ONNX Runtime, CPU, 4 threads, 200 iterations)
| Stage | Mean | p95 |
|---|---|---|
| Background-subtraction detection | 5.4 ms | 9.3 ms |
| Network (batched crops) | 5.8 ms | 8.4 ms |
| IK + trajectory planning | 0.8 ms | 1.1 ms |
| **Total** | **12.0 ms** | **17.5 ms** |

Measured while a SAC training job was using other cores (an idle PC measured as low as 4.3 ms), so treat it as an upper bound.
The Jetson's CPU path (11.25 ms) is comparable to this loaded PC; TensorRT on the Jetson GPU is about twice as fast as either.

## 5. SAC reach policy (`scripts/train_sac_reach.py`)
Task: drive the gripper fingertip to a random reachable target (full yaw, 5-30 cm height) within a **1 cm** radius; 17-d observation (joint state, fingertip, target, target - fingertip),
4-d joint-velocity action, 150 steps per episode. 300,000 training steps on CPU (about 1.5 hours, shared with other jobs). The earlier pick-and-place SAC task never succeeded and is replaced by this one.

| Metric (best checkpoint, 100 fresh episodes) | Measured |
|---|---|
| Success within 1 cm | **16%** (target not met) |
| Final fingertip error, mean / 95th percentile | 18.4 mm / 28.0 mm |
| Steps to success (successful episodes only) | 69 of 150 |

The analytic IK used by the sorting pipeline reaches targets in < 1 mm, so SAC is not on the critical path. A first version without the target-minus-fingertip observation plateaued
at about 16 cm average distance and was stopped at 150k steps. Reward was still improving at 300k steps; a longer run (1-2M steps, e.g. on Colab) is the obvious next experiment, not done here.

## 6. Planning and tests
- Analytic FK matches PyBullet on 50 random joint configurations; IK reaches 300 random targets to < 1 mm (PyBullet FK); every target the IK refuses lies inside the elbow-limit dead zone.
- 2000 random picks over the pick region x 4 classes: 0 infeasible.
- `python -m unittest discover tests`: 21 tests pass (kinematics vs PyBullet, trajectory incl. node-default sequencer, calibration, ONNX parity, detector geometry, demo-view helpers, 3-trial sorting, legacy env).

## 7. Not done / open
- ROS 2 nodes under a live ROS, ESP32 sketch compile and flash, the micro-ROS link, hardware e-stop wiring, `steps_per_rad` calibration: all still unverified. (The Jetson TensorRT engine, ONNX Runtime and sorting simulation are now measured, section 4.)
- Jetson sustained test was only 60 s per backend, in MAXN_SUPER mode; longer soak tests and 15 W / 25 W comparisons for the TensorRT path were not run.
- Capsule-like OOD detection; the detector missing some unseen meshes (5 of 30 in the unseen-object test); false e-stops near the 0.75 threshold (about 1% of single-object crops).
- SAC does not reach the 1 cm target (16%).
- The simulated grasp is a weld, not a physical grip; domain gap to the real IMX219 camera is untested.
- Everything was trained locally on CPU: perception takes about 10 minutes uncontended, and SAC's bottleneck is the CPU simulator, so Colab was not used. The original `train_colab.ipynb` still targets the removed pick-and-place task.
