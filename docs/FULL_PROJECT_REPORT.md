# Automator: Full Project Report

**Project:** Shape-sorting 4-DOF robotic arm, simulated in PyBullet, with the perception/control stack designed to run on a Jetson Orin Nano (ROS 2 Humble, TensorRT) and an ESP32 (micro-ROS) actuator/safety controller.
**Developer:** Siddhant Kumar (VIT Vellore). **Report scope:** the original project state, every experiment run during the rework, and the final results.

**How to read the numbers.** Every figure marked *measured* was produced by scripts in this repository, on one of two machines: the development PC (Arch Linux, Intel i7-1360P, 16 threads, **CPU only**, Python 3.14) or the **Jetson Orin Nano Super** (JetPack 6 / L4T R36.5, TensorRT 10.3, Python 3.10), as labelled. PC results are stored in `models/*.json`; Jetson results in `docs/jetson_results/` and `docs/JETSON_RESULTS.md`. Figures from earlier iterations are labelled *superseded* and are kept because they show what each change fixed. Figures quoted from the original `report.md` are labelled *original claim*. **The ROS 2 nodes and the ESP32 have not been run** (the arm is not built; only the simulation ran on the Jetson). Items written but never run on target hardware are marked *unverified*.

---

## Contents
1. Executive summary
2. Starting point: what the original project claimed vs. what it did
3. Final system design
4. Data: generation and statistics
5. Perception model: training procedure, experiments and final results
6. Planning and kinematics
7. End-to-end sorting results (and how they evolved)
8. Safety behaviour (OOD e-stop) results
9. SAC reinforcement learning
10. Latency / edge performance
11. Firmware, ROS 2 and Jetson status
12. Test suite
13. Debugging log: problems found and root causes
14. Targets scorecard
15. Limitations and open items
16. Reproduction commands and file map

---

## 1. Executive summary

| Area | Final result (measured) |
|---|---|
| End-to-end sorting, 4 objects, 20 trials | **74 / 80 = 92.5 %** in the correct bin, 0 e-stops |
| Unseen object on the table, 30 trials | e-stop in **25 / 30** trials (all on the unseen object); detector never saw it in the other 5; arm moved an unseen object in 1 / 30 |
| Shape classification (2,217 held-out crops) | **99.91 %** (2 errors) |
| Position error (held-out crops) | **1.97 mm** mean, **4.17 mm** p95 (detector geometry alone: 5.43 mm) |
| OOD AUROC, unseen procedural meshes | **0.999** |
| OOD AUROC, held-out capsule shape | **0.714** (13.6 % flagged) |
| OOD AUROC, all unseen objects combined | **0.934** (target ≥ 0.95: **not met**) |
| OOD false-positive rate at 95 % in-dist acceptance | **15.6 %** (target ≤ 10 %: **not met**) |
| SAC reach, 1 cm success radius | **16 %** success, 18.4 mm mean final error (**not met**) |
| Perception→plan latency, PC CPU (ONNX Runtime) | **12.0 ms** mean, 17.5 ms p95 |
| Perception→plan latency, **Jetson Orin Nano Super** | **5.9 ms** TensorRT FP16 (p95 6.2 ms), 6.2 ms FP32, 11.3 ms ONNX Runtime CPU; GPU compute alone 0.89 ms |
| Jetson sustained run (60 s at 30 Hz) | TensorRT FP16: 6.0 ms, no drift, 7 % CPU, 7.0 W; CPU path: 10.2 ms, 44 % CPU, 8.5 W |
| Sorting run on the Jetson (TensorRT FP16) | **12 / 12** objects, 3 H.264 videos |
| Automated tests | **21 / 21 pass** |
| ROS 2 nodes under live ROS / ESP32 / micro-ROS link | **not run** |

Headline lessons: the original model was trained on random-noise images and its reported results were not supported by its own tests; after rebuilding the data pipeline, the detector, the model calibration and the control stack, simulated sorting works at 92.5 %, arbitrary unseen objects are almost always caught, and the main remaining weakness is an unseen object that resembles a known class.

---

## 2. Starting point: original claims vs. original behaviour

### 2.1 What the original `report.md` claimed (*original claim*)
- End-to-end pipeline latency 10.28 ms (97.3 Hz); perception 9.90 ms; planner 0.29 ms; physics step 0.09 ms.
- 9 / 9 tests passing; E-stop precision 100 % with in-distribution uncertainty 0.0000 and OOD uncertainty 9,822.
- Verified non-slip pick → grasp → transport → place for sphere, cube and cylinder.
- Trial history: trajectory RMSE 0.477 m (heuristic velocity) → 0.082 m (default IK) → 0.031 m (high-precision IK, 1 mm surface clearance), then gripper and quintic planner trials.

### 2.2 What the original code actually did (*measured* when the baseline suite was run before any change)
- Baseline `unittest discover tests`: 9 tests OK, but the multi-object pick-and-place test printed **placement RMSE 0.3630 m and success rate 0.0 %**; it passed only because its assertion threshold was 0.40 m.
- The E-stop test printed in-distribution uncertainty **0.0000** (it fed the class mean itself, so distance is exactly 0) and OOD uncertainty **10,099.86** (an unbounded Mahalanobis distance compared with 0.75).
- The "perception" training data (`scripts/download_dataset.py` synthetic fallback) was **random-noise images with one red square**; the class label was `i % 5`, unrelated to the image; xyz was a function of the square's pixel position. Training ran 2 epochs on 50 such images.
- `evidential_vision_node` used random weights, fitted Mahalanobis statistics on `torch.randn` dummies, loaded no checkpoint or engine, fed BGR frames to an RGB model with no normalisation.
- `models/mobilenet_uncertainty.engine` was a **27-byte placeholder file** (`TRT_ENGINE_PLACEHOLDER_BIN`), written when TensorRT was absent.
- Firmware: no micro-ROS at all; `loop()` fed itself a hard-coded target; the "hardware ISR" was an ordinary function call; `Serial.printf` calls would corrupt the micro-ROS serial transport; the motor-enable pin was GPIO 12 (flash-voltage boot strap).
- The "latency" figure summed three unrelated timings (planner + random-weight 224×224 dummy inference + a PyBullet physics step).
- `scripts/train_rl_agent.py` used `EvalCallback` without importing it (NameError once stable-baselines3 was present).
- URDF elbow limit ±2.0 rad makes any target within ~0.30 m of the shoulder unreachable, yet objects spawned at r = 0.20–0.35 m.

A full 40-item list with fixes is in `docs/PROBLEMS_AND_FIXES.md`.

---

## 3. Final system design

### 3.1 Data flow
```
PyBullet scene (sim_bridge_node)                         Jetson Orin Nano
 floor, 4 trays, N objects, arm   --/camera/image_raw-->  evidential_vision_node
                                  --/camera/reference-->   1 background subtraction -> blobs
                                                           2 masked 128x128 crop per blob
                                                           3 MobileNetV3-small (TensorRT/ONNX):
                                                             shape logits, xyz residual, OOD score
                                                      /arm_target [x,y,z,class,uncertainty]
 SimArm (position control) <--/joint_commands-- joint_trajectory_node
                           <--/estop----------   analytic IK + quintic joint trajectories,
                                                 bin chosen by class, u>0.75 -> halt, /arm_busy
                                                      | /joint_commands /arm_target /estop /estop_reset
                                                 ESP32 (micro-ROS, serial): stepper trapezoids,
                                                 gripper servo, GPIO e-stop interrupt, 200 ms watchdog
```

### 3.2 Key design decisions
- **Classical detector + CNN classifier.** A fixed camera lets background subtraction find objects of *any* shape, so unknown objects are never silently ignored. The CNN classifies one masked crop and reports whether it recognises it.
- **Geometry first, learning second.** Blob centroids are back-projected onto the table plane (detector geometry alone: 5.43 mm mean error). The CNN adds a learned residual that uses crop and bounding-box features, bringing error to 1.97 mm.
- **Calibrated uncertainty.** Minimum Mahalanobis distance to the four class means (tied covariance), mapped to [0, 1] so that 0.75 equals the 99th percentile of held-out in-distribution distances (raw distance 21.11). The computation lives inside the exported graph.
- **Analytic IK** (numpy only) instead of PyBullet IK, so the Jetson needs no physics engine for planning.
- **Workspace restricted** to where the ±2.0 rad elbow limit allows IK: pick region r 0.30–0.38 m, |θ| ≤ 0.7 rad; bins at r = 0.33 m, angles 0.95/1.45/1.95/2.45 rad. IK feasibility checked over 2,000 random picks × classes: 0 infeasible.
- **Fixed camera** (replacing a decorative marker 0.8 m away giving ~17 px objects): eye (0.15, 0.40, 0.45), target (0.32, 0, 0.03), 55° vertical FOV, 640×480 (~1.3 px/mm). Shared by dataset, demo and Jetson node through `perception/camera.py`.
- **Shared configuration.** Tool length (0.09 m) and the four bin positions live in `arm_perception/config.py`, used by the node, demo, tests and simulator.

### 3.3 Object set and bins
Four sortable shapes with randomised size (±15 %): sphere, cube, cylinder, brick (elongated box). Colour is randomised independently of class so it cannot be a shortcut. Each shape has its own shallow rimmed tray (inner radius 0.065 m, rim height 0.025 m).

---

## 4. Data

### 4.1 Generator (`scripts/generate_sim_dataset.py`)
- In-distribution scenes: 1–4 objects per scene from the 4 classes; random colour, size, yaw, position (pick region), one of 8 light directions; trays in view.
- OOD scenes: unseen objects only: random procedural meshes from `pybullet_data/random_urdfs` (pool 0–499 for validation, pool 500–999 for test; scaled to ≤ 5.5 cm and dropped/settled) plus a held-out capsule primitive (25 % of OOD scenes).
- Each detected blob becomes one sample: 128×128 masked crop, bbox features, first-pass geometric xyz, ground-truth object centre (matched through the segmentation mask; blobs with < 70 % purity are dropped as merged).
- Splits are by scene.

### 4.2 Datasets produced (*measured*)
| Dataset version | Scenes | Samples | Notes |
|---|---|---|---|
| Mini (smoke test) | 40 + 10 | 110 | pipeline check; detection recall 110/126 at 0.09 m separation |
| v1 | 3,000 ID + 800 OOD | 5,385 | pick region r 0.27–0.35; recall 5,385/5,413; train 2,845 / val 578 / test 577; OOD val 688 / test 697 |
| v2 | 6,000 ID + 1,000 OOD | 16,495 | same layout, more data; recall 16,495/16,795; train 10,302 / val 2,139 / test 2,216; OOD 928 / 910 |
| v3 | 6,000 + 1,000 | 16,466 | new layout (r 0.30–0.38); unmasked crops, no trays in data; train 10,289 / val 2,134 / test 2,217; OOD 920 / 906 |
| **v4 (final)** | 6,000 + 1,000 | **16,464** | new layout, **masked crops, trays in scene**; recall 16,464/16,795 (98.0 %), 109 merged/unmatched blobs dropped; train **10,289** (sphere 2,599 / cube 2,568 / cylinder 2,598 / brick 2,524), val **2,134**, test **2,217**, OOD val **918**, OOD test **906** |

Generation time: ~53 s (3,800 scenes) to ~250 s (7,000 scenes) on 10–14 worker processes; rendering is ~0.14 s per 640×480 frame (TinyRenderer).

Dataset changes along the way: pick region moved outward (v2→v3) after IK rejected objects at r < 0.29 m; separation raised 0.09 → 0.12 → 0.11 m and the position sampler made restart-based after greedy sampling failed to fit 3–4 objects; crops became masked and trays were added (v3→v4) after a measured domain-shift problem (§13).

---

## 5. Perception model

### 5.1 Architecture
MobileNetV3-small features (ImageNet-pretrained) → global pool → 128-d embedding (ReLU) → heads: 4-way shape classifier; xyz residual regressor (embedding + 4 bbox features → 64 → 3, output in units of 1 cm). ImageNet normalisation is inside the model. The Mahalanobis statistics are buffers fitted after training. Exported graph `(crop, bbox_feat) → (logits, offset, uncertainty)` with dynamic batch (`models/perception.onnx`, 4.1 MB).

### 5.2 Training recipe (final)
AdamW (lr 1e-3, weight decay 1e-4), OneCycle schedule, batch 64, **20 epochs**, label smoothing 0.05; loss = cross-entropy + 0.5 × SmoothL1(offset in cm). On-GPU/CPU augmentation: brightness/contrast ±20 %, random colour-channel permutation, scale ±8 %, translation ±4 %, Gaussian noise σ 0.02; no flips (offsets are world-frame). After training: switch to eval mode, fit Mahalanobis means and tied inverse covariance (ridge 1e-3 × mean variance) on **clean** train features, calibrate the 0.75 threshold on validation features (99th percentile), evaluate on untouched test and OOD splits. Training loss (final run) fell from 0.742 (epoch 1) to 0.204 (epoch 16). Run time on CPU: ~10–11 min uncontended (~32–47 s/epoch); the final run took ~50 min because other jobs shared the CPU.

### 5.3 Experiment history (*measured*; earlier rows are superseded)
| Run | Data | Epochs | Shape acc. | xyz mean / p95 (mm) | False e-stop @0.75 | OOD AUROC (test) | Mesh AUROC | Capsule AUROC | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Smoke | v1 | 2, no pretrain | 25.4 % (predicts one class) | 4.50 / 8.84 (geometry 5.16) | 0.2 % | 0.71 | 0.82 | 0.43 | proves the pipeline end-to-end |
| A | v1 | 12, pretrained | 100 % (n = 577) | 2.03 / 4.10 | 1.4 % | 0.837 | 0.999 | 0.425 | 107 s of CPU training |
| B | v2 | 20, pretrained | 100 % (n = 2,216) | 2.01 / 4.98 | 1.13 % | 0.906 | 1.000 | 0.587 | |
| C | v3 | 20, pretrained | 100 % (n = 2,217) | 1.84 / 4.58 | 1.08 % | 0.926 | 1.000 | 0.677 | new layout; test OOD det. rate 81 % |
| Center-loss λ = 0.3 | v3 | 20 | **25.4 % (collapsed)** | 3.86 / 9.02 | 0 % | 0.50 | 0.50 | 0.50 | features collapsed to zero; calibrated distance 0; **dropped** |
| **D (final)** | **v4** | **20, pretrained** | **99.91 % (n = 2,217)** | **1.97 / 4.17** | **0.72 %** | **0.934** | **0.999** | **0.714** | masked crops + trays |

(An intermediate "center-loss 1.0" run on v2 data was killed at epoch 1 with loss 8.79 and never evaluated; a pair of parallel center-loss runs was killed for CPU contention before finishing, and a third perception job was OOM-killed on the 15 GB machine, which is why runs were later queued sequentially.)

### 5.4 Final results in detail (`models/perception_metrics.json`, *measured*)
**Calibration:** 99th percentile of validation in-distribution Mahalanobis distances = **21.108** ↔ uncertainty 0.75.

**In-distribution test (2,217 crops):**
- Shape accuracy **99.91 %**. Confusion (rows = true, columns = predicted; classes sphere, cube, cylinder, brick):

| | sphere | cube | cylinder | brick |
|---|---|---|---|---|
| sphere | 563 | 0 | 0 | 0 |
| cube | 0 | 588 | 0 | 1 |
| cylinder | 0 | 0 | 510 | 1 |
| brick | 0 | 0 | 0 | 554 |

- xyz error with refiner: mean **1.97 mm**, p95 **4.17 mm**; xy only: 1.51 mm. Detector geometry alone: 5.43 mm mean, 10.98 mm p95, 3.50 mm xy → the learned residual cuts mean error by 64 %.
- False e-stop rate at 0.75: **0.72 %**.

**OOD (unseen objects):**
| Split | n | AUROC all | Detected at 0.75 | FPR @ 95 % TPR | Meshes: AUROC / detected | Capsule: AUROC / detected |
|---|---|---|---|---|---|---|
| OOD val | 918 | 0.931 | 76.4 % | 16.2 % | 0.9987 / 97.1 % | 0.728 / 14.7 % |
| **OOD test** | **906** | **0.934** | **76.8 %** | **15.6 %** | **0.9985 / 95.4 %** | **0.714 / 13.6 %** |

Interpretation: arbitrary unseen objects are caught reliably; an unseen shape that looks like a known class (the capsule next to the cylinder class) mostly is not. The combined AUROC target is missed because the capsule is ~25 % of the OOD set. The only attempt to tighten the embedding (center loss) collapsed.

---

## 6. Planning and kinematics (*measured*)

- **Geometry** from the URDF: shoulder at z = 0.25 m; links 0.30 m and 0.25 m; yaw about z; three pitch joints; fingertips 0.09 m beyond the wrist.
- **IK** closed-form; elbow-forward then mirrored solution; clamped to URDF limits (|q1| ≤ π, |q2| ≤ 1.5708, |q3| ≤ 2.0, |q4| ≤ 1.5708); raises `IKError` if unreachable.
- **Verification against PyBullet FK:** analytic FK equals PyBullet on 50 random joint configurations (< 1e-6 m); IK reaches 300 random pick-region targets to < 1 mm; every refused target lies within 0.31 m of the shoulder (elbow-limit dead zone).
- **Reachability map (wrist target, before restricting the workspace):** with the original r = 0.20–0.35 m spawn range, targets at r ≤ 0.22 m were unreachable at every tested height and z = 0.25 m was unreachable for r ≤ 0.30 m (e.g. z 0.05: r 0.25 ok, 0.20 and 0.22 not; z 0.11: ok from r 0.30; z 0.25: ok only at 0.35). This drove the workspace restriction.
- **Sequencer:** waypoints approach (tip +0.05 m) → grasp (+0.01 m) → close → lift → transport → release (+0.005 m) → open → retreat → home; joint-space quintic segments (zero velocity/acceleration at both ends), duration = max(1 s, time to respect 80 % of URDF velocity limits using the quintic peak factor 1.875). Lift height = lowest IK-feasible z above both pick and bin (found: 0.15 m for the tested pair). A full 4-object pick cycle in simulation takes ≈ 11 s of simulated time.
- **Floor safety:** fingertip height stays > 0.02 m on every tested path.

---

## 7. End-to-end sorting results

### 7.1 Evolution (*measured*; each row changed one thing)
| Stage | Change | Result |
|---|---|---|
| First end-to-end run (1 trial, 3 objects) | model run B, old pick region, flat bins, wrist targets | 2 / 3 sorted; 1 left on table (IK "violates joint limits") |
| 8-trial diagnostic, old pick region | objects at r < 0.29 m | most failures were IK rejections at approach/grasp waypoints |
| After moving pick region and bins outward (model run C), flat bin discs, release 5 cm up | 20 trials × 4 objects | **35 / 80 = 43.8 %**; objects rolled out of bins (no rolling friction), 2 e-stops |
| + rolling friction/damping, rimmed trays, release 0.5 cm above floor | 10 trials × 4 | **32 / 40 = 80.0 %**; 4 of 10 trials ended in an e-stop on the last object |
| + masked crops, trays in training data (final model D) | 20 × 4, "least-uncertain first" policy | **73 / 80 = 91.2 %**; 4 trials ended in e-stop |
| + node-consistent policy (any OOD object stops the arm) | 20 × 4 | **70 / 80 = 87.5 %**; 3 trials ended in e-stop |
| + node-faithful loop (vision only when idle/settled) and image-space separation of spawned objects (**final**) | 20 × 4, seeds 1000–1019 | **74 / 80 = 92.5 %**, 0 e-stops |

A physics diagnosis explains the 43.8 %→80 % jump: the carried object tracked the fingertip correctly, but after release it landed tilted, tipped and **rolled indefinitely** (PyBullet default rolling friction is zero).

### 7.2 Final result (`models/sort_results.json`)
- 20 trials × 4 known objects: **74 / 80 correct**, 0 trials ended in an e-stop.
- Failures (6, all "ended outside own tray"): sphere ended ~10 cm from its tray (trial 4), cube 10.7 cm from its tray (trial 8), a brick in the sphere tray (trial 13 — a genuine misclassification) plus a sphere 11 cm away (trial 13), cube 9.8 cm away (trial 15), brick 11 cm away (trial 17). Causes other than the one misclassification were not isolated (candidates: knocked by the arm, bounced off a rim, dropped early).
- Demo video of trial 0: `models/sort_demo.mp4` (875 frames).

### 7.3 Simulation fidelity caveats
The grasp is a fixed constraint created at the object's current pose when the gripper closes within 4.5 cm of the fingertips; the finger geometry does not physically hold objects. Sorting success therefore tests perception, IK, trajectories and release/placement physics, not grasp mechanics.

---

## 8. Safety behaviour (OOD e-stop)

### 8.1 Unseen-object test (`models/sort_unknown_results.json`, seeds 5000–5029)
30 trials, each with 3 known objects and 1 unseen random mesh:
- **e-stop in 25 / 30 trials, all 25 on the unseen object, 0 on a known object.**
- The unseen mesh started inside the pick region in 27 of 30 trials; vision saw it in 25 trials and flagged it in all 25.
- In the other 5 trials the detector never saw the unseen object (background-subtraction miss, likely low contrast or small/flat meshes — not isolated); the arm sorted the known objects and moved an unseen object in 1 / 30 trials.
- (An earlier version of the same test with a slightly different decision policy gave 26 / 30 e-stops and 1 / 30 unseen objects moved.)

### 8.2 Diagnostics behind the safety design
| Experiment | Finding |
|---|---|
| Single-object scenes, 300 samples each, without / with trays (unmasked model) | fraction u > 0.75: **0.3 %** vs **2.0 %**; for θ 0.35–0.7 it rose from 0 % to **7.5 %** with trays → trays/neighbours in the crop context caused domain shift → fixed by masked crops + trays in data |
| Single-object scenes, final model, 400 samples | overall **0.5 %** false alarms (consistent with 99th-percentile calibration); 1 of 23 near a tray; per class: sphere 0 %, cube 0 %, cylinder 1 %, brick 1.1 % |
| 4-object scenes, 150 scenes, 590 blobs | **10 blobs contained > 1 object and all 10 were flagged**; single-object blobs 6 / 580 (1.0 %) flagged → merged silhouettes drove many "false" e-stops → demo now spawns objects ≥ 100 px apart in the image |
| Vision running during arm motion (3 sampled runs, 49 / 46 / 54 checks at 5 Hz) | u > 0.75 in 0 / 0 / 4 checks (arm's own links and held object crossing the pick area) → vision now pauses while `/arm_busy` + 0.6 s settle; tradeoff: an unknown object entering mid-motion is caught only when the arm stops |

---

## 9. SAC reinforcement learning

### 9.1 Original task (superseded)
Pick-and-place with `-distance` reward, 3 object types, 18-d observation. Never succeeded (0.0 % in the integration test, RMSE 0.363 m). The original Colab runs (best checkpoints `models/best_model.zip`, `best_sac_policy.zip`) were not re-evaluated; the training script could not run as written (missing `EvalCallback` import). The architecture document's "14-d state / 4-d action" also disagreed with the 18-d / 5-d code.

### 9.2 New task: fingertip reach, 1 cm success radius (`sim/reach_env.py`, `scripts/train_sac_reach.py`)
Reach a random target (full yaw, r 0.27–0.36 m, z 0.05–0.30 m, kept only if IK-reachable); episode 150 steps at 60 Hz control; action = 4 joint velocities (× 2 rad/s); SAC (MlpPolicy 256×256, lr 3e-4, buffer 300k, batch 256, γ 0.98, learning_starts 5,000, auto entropy), evaluated every 10k steps on 20 episodes.

| Version | Observation / reward | Outcome (*measured*) |
|---|---|---|
| v1 | 14-d (no target–tip delta); reward −dist − 0.01·‖a‖² + 10 on success | evaluation reward −99.8 (10k steps) → −24.1 (150k steps), plateaued at ≈ 16 cm average distance; **stopped at 150k steps** |
| **v2 (final)** | 17-d (adds target − fingertip); reward −10·dist − 0.01·‖a‖² + 1 inside 3 cm + 20 on success | evaluation reward −586 (10k) → −259 (30k) → −146 (100k) → −104 (230k) → −116 (260k); finished 300k steps |

Final evaluation of the best v2 checkpoint, **100 fresh episodes** (`models/sac_reach/metrics.json`):
- success within 1 cm: **16 %** (target not met);
- final fingertip error: mean **18.4 mm**, p95 **28.0 mm**;
- steps to success (successful episodes only): 69 of 150.

Throughput: ~3,200 simulator steps/s without learning; SAC training ran at 59–82 steps/s on 4 threads while sharing the CPU (~1.5 h for 300k steps). Reward was still improving at 300k steps; a 1–2 M step run is the obvious next experiment (not done). The sorting pipeline does not depend on SAC: its IK reaches targets in < 1 mm.

---

## 10. Latency and edge performance

### 10.1 Final benchmark (`models/benchmark_dev_pc_onnx_cpu.json`, *measured*)
ONNX Runtime CPU, 4 threads, 200 iterations, real sim frames with 3 objects; measured while a SAC job used other cores (treat as an upper bound).

| Stage | Mean | p50 | p95 | Max |
|---|---|---|---|---|
| Background-subtraction detection | 5.39 ms | 5.31 | 9.29 | 15.1 |
| Network (batched crops) | 5.77 ms | 5.49 | 8.44 | 11.4 |
| IK + trajectory planning | 0.80 ms | 0.70 | 1.09 | 2.9 |
| **Total** | **11.96 ms** | 11.62 | **17.45** | 20.0 |

### 10.2 Benchmark history
| Version | Detection | Total |
|---|---|---|
| First benchmark (before optimisation, under heavy load) | 31.3 ms | 36.9 mean / 60.8 p95 |
| After crop-window optimisation (no change to output) | ~9.3 ms standalone | |
| After replacing `absdiff().max(axis=2)` with `cv2.max` | **2.2 ms** standalone (same 4 blobs) | final table above under load |

(*Original claim*, superseded: 10.28 ms total, built from a random-weight model on a dummy 224×224 input, a planner timing and a physics step.)

### 10.3 Jetson Orin Nano Super (*measured* on the device; `docs/JETSON_RESULTS.md`)
Jetson Orin Nano Engineering Reference Developer Kit Super, JetPack 6 (L4T R36.5), 6 cores, 7.4 GB, TensorRT 10.3.0, ONNX Runtime 1.23.2 (CPU only), power mode MAXN_SUPER with `jetson_clocks` unless stated.
Per frame, 3 objects, 500 iterations each, run through the web dashboard:

| Backend | Detect | Network | Plan | **Total mean** | p95 | max |
|---|---|---|---|---|---|---|
| ONNX Runtime, CPU | 4.14 ms | 5.94 | 1.17 | **11.25 ms** | 13.84 | 27.61 |
| TensorRT FP32 | 3.04 | 2.04 | 1.08 | **6.16 ms** | 6.48 | 6.80 |
| **TensorRT FP16** | 3.00 | 1.84 | 1.07 | **5.91 ms** | 6.20 | 6.83 |

- TensorRT cuts the network step 3.2x and the whole path 1.9x, and removes CPU jitter (worst case 27.6 ms to 6.8 ms). FP16 outputs match the FP32 CPU model within 0.005 in the logits with identical classes.
- `trtexec` GPU compute (batch 3): FP16 **0.892 ms** (1117 qps), FP32 1.096 ms (910 qps); batch 1: 0.780 / 0.919 ms. The network step in the pipeline is about twice the GPU time (host-device copies, Python).
- Power mode: in the original 25 W mode the CPU path took **14.02 ms** (p95 14.93) vs 11.0 ms in MAXN_SUPER.
- Sustained, 60 s at 30 Hz (about 1,800 frames): TensorRT FP16 total 6.03 ms (p95 6.31, max 6.72, first half 6.01 vs second half 6.04), GPU 2 %, CPU 7 %, **7.0 W** avg (7.7 max), 48 C; ONNX CPU 10.17 ms (p95 10.63, max 17.84), CPU 44 %, 8.5 W, 47 to 49 C. No throttling in 60 s; a longer soak was not run.
- Engine build on the Jetson: FP16 195 s, FP32 85 s.
- Sorting run with TensorRT FP16, 3 trials x 4 objects (seeds 1000-1002): **12 / 12** sorted, 0 e-stops, 3 videos (about 650 kB each, H.264). It took 711 s of wall time because the simulator is CPU-bound on the Jetson (simulated camera render 270-350 ms/frame); perception inside the run was about 5.6 ms (detect 3.3 + network 2.36).
- The original report's projected "about 1.2 ms INT8" figure was never measured and is not supported.
- Setup findings (all fixed in `scripts/jetson_install.sh`, see `docs/PROBLEMS_AND_FIXES.md` G): NumPy 2 breaks the apt OpenCV; a fresh image has no TensorRT/CUDA and needs the DLA libraries plus `ldconfig`; no CUDA PyTorch on the board, so the TensorRT backend uses `cuda-python`; `ffmpeg` is needed for browser-playable video.

---

## 11. Firmware, ROS 2 and tooling status (ROS and ESP32 *unverified*)

| Component | State |
|---|---|
| `joint_trajectory_node` | written; logic (sequencer) unit-tested; node not run under ROS |
| `evidential_vision_node` | rewritten (TensorRT/ONNX/torch backends, rgb8/bgr8 + stride handling, batching, rate limit, `/arm_busy` pause); not run under ROS |
| `sim_bridge_node` | written (PyBullet camera → ROS, `/joint_commands` → arm, respawn, `inject_unknown`, `/sim/spectator`, optional PyBullet GUI); not run |
| `sim_viewer_node`, `overlay.py`, `launch/sort_demo.launch.py` | viewer window (simulation \| annotated perception, status bar, `e`/`r` keys, optional mp4), one-command launch; overlay helpers unit-tested, nodes not run under ROS |
| Edge-test dashboard (`webapp/`, `scripts/start_dashboard.sh`) | **run on the Jetson**: all job types exercised (self-check, benchmarks, live latency, trtexec, sorting with video), downloads, zip, range requests, token auth and path-traversal / injection checks |
| `/esp32/status` heartbeat in the sketch | written, **not compiled**: bit flags for e-stop, motors, fresh command, hardware line, plus a received-command counter |
| ROS packaging | `resource/` marker and `setup.cfg` added, `AUTOMATOR_ROOT` path resolution; needs `colcon build` on the Jetson |
| ESP32 sketch (`firmware/esp32_microros/esp32_actuator/`, Arduino IDE + micro_ros_arduino) | written, **not compiled**; micro-ROS in a core-0 task, AccelStepper on core 1; real GPIO33 interrupt e-stop; 200 ms watchdog (signed-difference race fixed); latched e-stop with reset topic; no `Serial` prints; GPIO12 avoided |
| Stepper calibration | `steps_per_rad` are **placeholders** (assumed gear ratios); must be measured before powering the arm |
| Known firmware limitation | disabling drivers removes holding torque (arm sags); no limit switches/encoders (assumes q = 0 at power-up) |

A PlatformIO build was attempted early (the micro-ROS library builder failed on a missing `penv` directory in the throw-away setup); it was abandoned when the user confirmed they use the Arduino IDE.

---

## 12. Test suite (*measured*: 21 tests, all pass)

| File | Tests | Covers |
|---|---|---|
| `test_kinematics.py` | 3 | analytic FK = PyBullet (50 configs); IK within 1 mm over 300 targets, refusals only in the elbow dead zone; unreachable target rejected |
| `test_trajectory.py` | 7 | joint limits and velocity limits respected, path returns home; gripper closes at the pick and opens at the bin; fingertips stay above the floor; high uncertainty halts and blocks until cleared; unreachable/busy rejection; halt freezes the setpoint; **node-default sequencer** (fingertip within 2 mm of object, each class released over its own bin, unknown class does not mutate another bin) |
| `test_perception.py` | 2 | forward shapes; fit + predict with uncertainty |
| `test_integration.py` | 4 (+) | calibration (≈ 1 % in-dist above 0.75, far OOD above it, bounded ≤ 1); ONNX parity at batch 3 vs batch-1 export; detector back-projection (< 6 mm with true height, < 12 mm with assumed height); end-to-end 3-trial sort (≥ 7 / 9 required; measured 8 / 9) |
| `test_overlay.py` | 3 | ESP32 status bit decoding, detection overlay drawing, viewer composition with missing images |
| `test_sim.py` | 2 | legacy environment |

Test-suite corrections made during the rework: the ONNX parity test initially failed because a random-initialised network on noise outputs near-constant values and because `torch.onnx.export` restores the wrapper's training flag (flipping BatchNorm back to training mode); verified separately that the real exported model matches PyTorch to ~1e-6 at batch 16 (max abs diff: logits 1.4e-6, offset 1.9e-8, uncertainty 2.7e-6; argmax identical).

---

## 13. Debugging log: root causes found while building

1. **IK dead zone** (elbow limit ±2.0 rad ⇒ targets within ~0.30 m of the shoulder unreachable) — mapped, then workspace restricted; later approach waypoints rejected objects at r < ~0.29 m, so pick region and bins were moved outward and approach/grasp heights lowered (0.08/0.02 → 0.05/0.01).
2. **Floor penetration:** planning the wrist (not fingertips) put fingers 9 cm below the target → fingertip-based planning (`TOOL_LENGTH = 0.09`).
3. **Rolling objects:** no rolling friction in PyBullet ⇒ released cylinders and spheres rolled out of bins → rolling friction 0.02, damping, shallow trays, lower release height.
4. **Grasp teleport:** original weld snapped objects to a fixed offset → weld at the current relative pose.
5. **Context leakage:** tray edges and neighbouring objects inside crops raised false e-stops (7.5 % in one image band) → masked crops, trays in the data.
6. **Merged blobs:** overlapping silhouettes always flagged OOD → image-space separation in the demo.
7. **Vision during motion:** arm links flagged as unknown → `/arm_busy` gating.
8. **Node/demo mismatch** (found in review): the ROS node planned for the wrist and had an empty class→bin map (all classes to one bin) → shared `config.py`, test that builds the sequencer exactly like the node.
9. **Detector speed:** per-blob full-frame padding/masking and a strided channel max dominated runtime → crop-window processing and `cv2.max` (detection 9.3 → 2.2 ms).
10. **Training infrastructure:** parallel jobs exhausted the 15 GB RAM (each perception run holds ~4 GB of tensors) and oversubscribed the CPU; runs were serialised. Several shell commands that used `pkill -f`/`pgrep -f` patterns appearing in their own command line killed the tool shell and had to be redone.
11. **SAC v1 plateau** (no target-minus-tip observation, reward scale 1) → v2 observation and reward shaping.
12. **Center-loss collapse** (features → 0, calibrated distance 0) → dropped.

---

## 14. Targets scorecard

| Target (chosen "strict" set) | Result | Met |
|---|---|---|
| Shape accuracy ≥ 99 % | 99.91 % | yes |
| xyz MAE ≤ 5 mm | 1.97 mm | yes |
| OOD AUROC ≥ 0.95 | 0.934 (meshes 0.999, capsule 0.714) | **no** |
| OOD FPR @ 95 % TPR ≤ 10 % | 15.6 % | **no** |
| Calibrated e-stop threshold | 99th percentile of validation; 0.72 % false e-stops | yes |
| SAC 1 cm success radius | 16 % success | **no** |
| Sort objects by shape in simulation | 92.5 % over 80 objects | yes |
| Unknown object triggers e-stop | 25 / 30 trials | partially (5 detector misses) |
| Phase 5 implemented (trajectory node, stepper driver, ESP32 coupling) | written | yes (unverified on hardware) |
| Perception + sorting on the Jetson | TensorRT FP16 5.9 ms, 12 / 12 sorted | yes |
| ROS 2 nodes live / ESP32 flashed and linked | not done | **no** |

---

## 15. Limitations and open items
- **Hardware:** ROS nodes under live ROS 2, ESP32 compile/flash and micro-ROS link, stepper calibration, hardware e-stop wiring, brakes/holding torque.
- **Jetson tests are short and single-board:** 60 s sustained runs, one power mode for TensorRT, one board; the simulator itself is far slower than real time on the Jetson CPU.
- **Capsule-like OOD:** an unseen object that looks like a known class is mostly accepted.
- **Detector misses:** 5 of 30 unseen meshes were never detected.
- **Concurrent-motion blind spot:** vision is paused during motion by design.
- **Merged silhouettes** stop the arm (conservative) — the demo avoids them by spacing.
- **Sort failures:** 6 of 80 objects ended outside their tray; one confirmed misclassification, the rest unexplained.
- **Simulation realism:** weld-based grasp; no real camera noise/lighting; domain gap to the physical IMX219 untested.
- **SAC** below the 1 cm target (longer training suggested).
- **Training environment:** CPU only; one random seed per configuration; no confidence intervals (small test counts for sorting: 80 and 90 objects).
- **Stale legacy files:** `train_colab.ipynb`, `scripts/record_*.py`, `scripts/render_simulation.py`, `scripts/demo_gui.py`, `sim/arm_env.py` and `sim/trajectory_planner.py` still target the old pick-and-place task.
- **Secrets:** `client_secrets.json` is now git-ignored; rotate it if it was ever shared.
- **Removed during rework** (backup tarball in the session scratchpad): Cornell dataset code/data/tests, `train_and_export_trt.py`, `benchmark_pipeline.py`, the placeholder engine, the old ONNX.

---

## 16. Reproduction and file map

```bash
.venv/bin/python scripts/generate_sim_dataset.py            # ~4 min, writes data/sim_shapes.npz
.venv/bin/python scripts/train_perception.py --pretrained   # ~10 min CPU; writes models/perception.{pt,onnx}, perception_metrics.json
.venv/bin/python scripts/sort_demo.py --trials 20 --objects 4 --video models/sort_demo.mp4
.venv/bin/python scripts/sort_demo.py --trials 30 --objects 4 --unknown 1 --seed 5000 --out models/sort_unknown_results.json
.venv/bin/python scripts/train_sac_reach.py --timesteps 300000 --out models/sac_reach
.venv/bin/python scripts/benchmark_edge.py --model models/perception.onnx --tag dev_pc_onnx_cpu
.venv/bin/python -m unittest discover tests
# on the Jetson: scripts/jetson_install.sh; scripts/build_trt_engine.sh fp16; scripts/start_dashboard.sh (browser) or
#   python3 scripts/benchmark_edge.py --model models/perception_fp16.engine --tag jetson_trt_fp16
```

| Path | Purpose |
|---|---|
| `perception/` | `detector.py`, `camera.py`, `mobilenet_uncertainty.py`, `inference.py` (TensorRT/ONNX/torch), `targeting.py` |
| `sim/` | `scene.py`, `sim_arm.py`, `reach_env.py`, URDF; legacy `arm_env.py`, `trajectory_planner.py` |
| `ros2_ws/src/arm_perception/arm_perception/` | `config.py`, `kinematics.py`, `trajectory.py`, `joint_trajectory_node.py`, `evidential_vision_node.py`, `sim_bridge_node.py`, `sim_viewer_node.py`, `overlay.py`, `launch/sort_demo.launch.py`, `imx219_camera_node.py` |
| `webapp/` | edge-test dashboard (`server.py`, `static/index.html`) |
| `scripts/` | dataset, training, sort demo, SAC, benchmark, TensorRT build, `jetson_install.sh`, `jetson_selfcheck.py`, `start_dashboard.sh`, `run_jetson_demo.sh`, `start_microros_agent.sh` |
| `firmware/esp32_microros/esp32_actuator/` | Arduino sketch, `pins.h`, `stepper_driver.{h,cpp}`; README with wiring and safety |
| `models/` | `perception.{pt,onnx}`, `perception_metrics.json`, `sort_results.json`, `sort_unknown_results.json`, `benchmark_dev_pc_onnx_cpu.json`, `sac_reach/`, `sort_demo.mp4` |
| `docs/` | `PROBLEMS_AND_FIXES.md` (40 items), `JETSON_RESULTS.md` + `jetson_results/` (Jetson measurements), this report |
| `report.md`, `architecture.md`, `README.md`, `task_plan.md` | concise report, design, usage, task log |
