# Jetson Orin Nano: measured results (2026-10-13)

All numbers were measured on the Jetson itself (run over SSH and through the web dashboard); raw files are in `docs/jetson_results/`.
Not covered: the ROS 2 nodes and the ESP32 (not run), and any real camera.

## Device and software
Jetson Orin Nano Engineering Reference Developer Kit **Super**, JetPack 6 (L4T R36.5), Python 3.10.12, 6 CPU cores, 7.4 GB RAM.
TensorRT 10.3.0 + CUDA 12.6 runtime (installed during this session), ONNX Runtime 1.23.2 (CPU only: no GPU provider on aarch64 PyPI), NumPy 1.26.4.
Power mode: **MAXN_SUPER with `jetson_clocks`** unless stated; the board was originally in 25 W mode and was restored to it afterwards.

## 1. Perception → planning latency per frame (3 objects per frame, 500 iterations, via the dashboard)
Stages: background-subtraction detection, network on a batch of crops (includes host↔device copies and Python overhead), IK + trajectory planning.
Simulator rendering is excluded (it is not part of a robot).

| Backend | Detect | Network | Plan | **Total mean** | p95 | max |
|---|---|---|---|---|---|---|
| ONNX Runtime, CPU | 4.14 | 5.94 | 1.17 | **11.25 ms** | 13.84 | 27.61 |
| TensorRT FP32 (GPU) | 3.04 | 2.04 | 1.08 | **6.16 ms** | 6.48 | 6.80 |
| **TensorRT FP16 (GPU)** | 3.00 | 1.84 | 1.07 | **5.91 ms** | 6.20 | 6.83 |

TensorRT cuts the network step 3.2× (5.94 → 1.84 ms) and the whole path 1.9×, and removes the CPU jitter (max 27.6 ms → 6.8 ms).
FP16 vs FP32 is a small difference for a network this small (5.91 vs 6.16 ms). FP16 outputs match the FP32 CPU model to within 0.005 in the logits, with identical predicted classes (batch 1, 3 and 8).

Earlier command-line runs (200 iterations) agree: TensorRT FP16 5.97 ms (p95 6.29), FP32 6.18 ms; ONNX CPU 11.01 ms (p95 11.89) in MAXN_SUPER and **14.02 ms (p95 14.93) in the 25 W mode** (CPU detect 4.75, network 7.71, plan 1.56).
For comparison, the development PC (i7-1360P, CPU only, ONNX) measured 12.0 ms in the earlier report.

## 2. Pure GPU time of the engine (`trtexec`, 300 iterations, MAXN_SUPER)
| Engine | Batch | GPU compute mean | p95 | Throughput |
|---|---|---|---|---|
| FP16 | 3 | **0.892 ms** | 0.895 | 1117 qps |
| FP32 | 3 | 1.096 ms | 1.100 | 910 qps |
| FP16 / FP32 | 1 | 0.780 / 0.919 ms | | |

The 1.84 ms the pipeline sees for the network step is about twice the 0.89 ms GPU time: the rest is host↔device copies, Python and the CUDA launch.
Engine build time on the Jetson: FP16 195 s, FP32 85 s.

## 3. Sustained live test (60 s each at 30 Hz, about 1,800 frames)
| Backend | Total mean | p95 | max | first vs second half | GPU avg | CPU avg | Power avg (max) | Temp |
|---|---|---|---|---|---|---|---|---|
| TensorRT FP16 | 6.03 ms | 6.31 | 6.72 | 6.01 → 6.04 | 2 % | 7 % | **7.0 W** (7.7) | 48 °C |
| ONNX Runtime CPU | 10.17 ms | 10.63 | 17.84 | 10.18 → 10.15 | 0 % | 44 % | 8.5 W (8.7) | 47 → 49 °C |

No latency drift or thermal throttling was seen over 60 s. TensorRT leaves the CPU almost idle (7 % vs 44 %) and draws less power while being faster.
(60 s is short: a longer soak would be needed to see thermal limits. Telemetry is `tegrastats`, sampled every 2 s.)

## 4. Sorting run on the Jetson (TensorRT FP16)
3 trials × 4 objects (seeds 1000-1002): **12 / 12 sorted correctly**, no e-stops. Three H.264 videos were produced (about 650 kB each, 640×480, 875-910 frames).
Perception cycles inside the run: detection 3.3 ms + network 2.36 ms (max 6.9) ≈ **5.6 ms**.
The run took 711 s of wall time because the *simulator* is CPU-bound on the Jetson (the simulated camera render alone is 270-350 ms per frame, plus physics and the spectator-view renders), not because of perception.
Consequence: on the Jetson set the simulated camera to 2-3 Hz (`camera_hz`), and do not expect the sim to run in real time with the spectator view.

## 5. What had to be fixed to get here (now in `scripts/jetson_install.sh`)
- `pip install onnxruntime` pulls **NumPy 2**, which breaks the apt OpenCV (`_ARRAY_API not found`): pin `numpy<2`.
- A fresh JetPack 6 image here had **no TensorRT or CUDA**. `apt install tensorrt python3-libnvinfer libnvinfer-bin` adds them (31 packages), but TensorRT then fails to load until
  `nvidia-l4t-dla-compiler` and `libcudla-12-6` are also installed and `sudo ldconfig` is run (`libnvdla_compiler.so` / `libcudla.so.1: cannot open shared object file`).
- There is no PyTorch CUDA wheel on this board, so the TensorRT backend was rewritten to use `cuda-python` instead of PyTorch.
- `ffmpeg` was not installed; it is needed to make the videos browser-playable (OpenCV writes MPEG-4 part 2, which browsers cannot play).
- ONNX Runtime from PyPI is CPU-only on aarch64; the GPU path here is the native TensorRT backend.

## 6. Reproduce
```bash
scripts/start_dashboard.sh                      # then use the browser: Compare all backends / Live latency test / Sorting run
python3 scripts/benchmark_edge.py --model models/perception_fp16.engine --tag jetson_trt_fp16
/usr/src/tensorrt/bin/trtexec --loadEngine=models/perception_fp16.engine --shapes=crop:3x3x128x128,bbox_feat:3x4 --iterations=300 --warmUp=500
```
