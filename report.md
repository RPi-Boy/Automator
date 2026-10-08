# Results report

Every number below was measured by the scripts in this repository on the development PC (Arch Linux, Intel i7-1360P, 16 threads, CPU only, Python 3.14),
and is stored next to the models in `models/*.json`. **Nothing here was measured on the Jetson Orin Nano or the ESP32.** Earlier versions of this
report quoted results (10.28 ms pipeline latency, clean pick-and-place) that the code did not support; see `docs/PROBLEMS_AND_FIXES.md`.

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

`models/sort_demo.mp4` shows trial 0 from a spectator camera.
Tradeoff: because perception pauses while the arm moves, an unknown object that appears mid-motion is only caught once the arm stops. (With vision running during motion,
1 of 3 sampled runs scored u > 0.75 from the arm's own links crossing the pick area, so the pause is necessary with a background-subtraction detector.)

## 4. Latency on the development PC (`scripts/benchmark_edge.py`, ONNX Runtime, CPU, 4 threads, 200 iterations, 3 objects per frame)
| Stage | Mean | p95 |
|---|---|---|
| Background-subtraction detection | 5.4 ms | 9.3 ms |
| Network (batched crops) | 5.8 ms | 8.4 ms |
| IK + trajectory planning | 0.8 ms | 1.1 ms |
| **Total** | **12.0 ms** | **17.5 ms** |

Measured while a SAC training job was using other cores, so treat it as an upper bound for this PC. **Jetson (TensorRT FP16) numbers are not measured yet:**
run `scripts/build_trt_engine.sh fp16` and `scripts/benchmark_edge.py --model models/perception_fp16.engine --tag jetson_fp16` on the device.

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
- `python -m unittest discover tests`: 18 tests pass (kinematics vs PyBullet, trajectory incl. node-default sequencer, calibration, ONNX parity, detector geometry, 3-trial sorting, legacy env).

## 7. Not done / open
- TensorRT engine, ROS 2 nodes under a live ROS, ESP32 sketch compile and flash, hardware e-stop wiring, `steps_per_rad` calibration: all unverified.
- Capsule-like OOD detection; the detector missing some unseen meshes (5 of 30 in the unseen-object test); false e-stops near the 0.75 threshold (about 1% of single-object crops).
- SAC does not reach the 1 cm target (16%).
- The simulated grasp is a weld, not a physical grip; domain gap to the real IMX219 camera is untested.
- Everything was trained locally on CPU: perception takes about 10 minutes uncontended, and SAC's bottleneck is the CPU simulator, so Colab was not used. The original `train_colab.ipynb` still targets the removed pick-and-place task.
