# Jetson Testing Instructions

**Audience:** juniors running the Automator demo on a Jetson Orin Nano to test how it performs at the edge.
**Scope:** only the **simulation** runs. There is no physical arm, so nothing here moves hardware. The simulated arm sorts objects by shape, and a window on the Jetson shows it. The **ESP32 is flashed and connected over USB** to test the micro-ROS link and the safety logic, with **no motors, drivers or arm attached**.

Read the whole page once before starting. Each step says what you should see; **if something differs, stop and look at [Troubleshooting](#7-troubleshooting)** rather than pressing on.

| Part | What you do | Time |
|---|---|---|
| 1 | Get the code, install, run the self-check | 30-40 min (pybullet compiles) |
| 2 | Simulation + perception without ROS (quick, safe baseline) | 10 min |
| 3 | TensorRT engine and speed measurements | 20 min |
| 4 | Full ROS 2 demo with the simulation window | 20 min |
| 5 | ESP32: flash, connect, test | 45 min |
| 6 | Collect results and send them back | 10 min |

---

## 0. What you need
- Jetson Orin Nano with JetPack 6 (Ubuntu 22.04), **ROS 2 Humble already installed** (`ls /opt/ros/humble` works), internet access, and a **monitor + keyboard + mouse** attached (the demo opens a window; plain SSH cannot show it).
- ESP32 DevKit (ESP32-WROOM-32) and a **USB data cable** (many cables are charge-only and will not work).
- A laptop with the Arduino IDE to flash the ESP32 (or use the Jetson itself).
- One jumper wire (for the hardware e-stop test).
- **Do not connect stepper drivers, motors or a power supply to the ESP32.** USB power only.

---

## Easiest way: the web dashboard
Most of Parts 2-4 below can be done from a browser. After Part 1 (install) run on the Jetson:
```bash
scripts/start_dashboard.sh
```
It prints a URL like `http://<jetson-ip>:8080/?token=XXXX`. Open it from any computer on the same network.
You can run the self-check, latency benchmarks (CPU vs TensorRT), a live latency chart, the TensorRT engine build, `trtexec`,
and sorting runs with video, all with live logs; then **download the results and videos** (single files or a zip per run).
Keep the token private: anyone with it can start tests on the Jetson. The ESP32 and ROS parts (Parts 4-5) still need a terminal.

---

## 1. Get the code and install

```bash
git clone https://github.com/RPi-Boy/Automator.git
cd Automator
export AUTOMATOR_ROOT=$PWD          # needed in EVERY new terminal (or add it to ~/.bashrc)
scripts/jetson_install.sh           # apt (TensorRT 10 + DLA libs, ffmpeg, OpenCV, micro-ROS agent) + pip (numpy<2, pybullet, onnxruntime, cuda-python); pybullet may compile: 10-20 min
```

Build the ROS package:
```bash
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --symlink-install --packages-select arm_perception && cd ..
source ros2_ws/install/setup.bash
```
`--symlink-install` matters: it lets the nodes find the `perception/` and `sim/` folders in this repository.

Run the self-check (every later step assumes it passes):
```bash
python3 scripts/jetson_selfcheck.py
```
**Expected:** a list of `[PASS]` lines and **`0 FAIL`** at the end. `[WARN]` for torch, the TensorRT engine (not built yet) and the ESP32 is fine at this point; `tensorrt` should import (version 10.x). Typical results to look at: `PyBullet render ... ms/frame` (should be well under 300), `detector finds 3 spawned objects`, `inference via onnxruntime[...]`. The file `jetson_selfcheck.txt` is written, which you will send back in Part 6.

**Every new terminal** needs:
```bash
cd ~/Automator && export AUTOMATOR_ROOT=$PWD && source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
```

---

## 2. Simulation and perception without ROS (baseline)
This isolates the perception/planning code from ROS, so if ROS gives you trouble later you still have numbers.

**Put the Jetson in full-speed mode first** (otherwise timings are meaningless):
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

**2a. Watch it sort (window):**
```bash
python3 scripts/sort_demo.py --trials 1 --objects 4 --gui
```
A PyBullet window opens showing the arm, 4 shape trays and the objects; the arm should pick each object and drop it in the tray for its shape. The console ends with `SORT SUCCESS: n/4 ...` (3 or 4 of 4 is normal; the reference result on the development PC was 92.5 % over 80 objects).

**2b. Measure speed (no window):**
```bash
python3 scripts/benchmark_edge.py --model models/perception.onnx --tag jetson_onnx
```
**Expected:** JSON printed, then `wrote models/benchmark_jetson_onnx.json`. The reference PC (Intel i7, CPU only) gave **detection 5.4 ms + network 5.8 ms + planning 0.8 ms = 12.0 ms mean** (`total.mean_ms`). Write your numbers down; they are the point of this exercise.
**Reference on a Jetson Orin Nano Super (MAXN_SUPER): ONNX Runtime CPU total 11.3 ms mean; in the default 25 W mode about 14 ms.** Much slower than this means the board is in a low power mode or is throttling.

---

## 3. TensorRT: build the engine and measure
Engines must be built **on this Jetson** (they do not transfer between machines).

```bash
scripts/build_trt_engine.sh fp16          # ~2-5 min  -> models/perception_fp16.engine
scripts/build_trt_engine.sh fp32          # for comparison -> models/perception_fp32.engine
```
**Expected:** `built models/perception_fp16.engine` (reference: FP16 about 3 minutes, FP32 about 1.5 minutes). If `trtexec` is not found: `export TRTEXEC=$(find /usr -name trtexec 2>/dev/null | head -1)` and rerun.

**3a. Pure GPU latency (needs nothing but TensorRT):**
```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=models/perception_fp16.engine \
   --shapes=crop:3x3x128x128,bbox_feat:3x4 --iterations=200 2>&1 | tee trtexec_fp16.log | tail -25
/usr/src/tensorrt/bin/trtexec --loadEngine=models/perception_fp32.engine \
   --shapes=crop:3x3x128x128,bbox_feat:3x4 --iterations=200 2>&1 | tee trtexec_fp32.log | tail -25
```
Note the lines `GPU Compute Time: ... mean = ... ms` for FP16 and FP32. **Reference (batch 3, MAXN_SUPER): FP16 about 0.89 ms, FP32 about 1.10 ms.**

**3b. Whole pipeline with the engine:**
```bash
python3 scripts/benchmark_edge.py --model models/perception_fp16.engine --tag jetson_trt_fp16
python3 scripts/benchmark_edge.py --model models/perception_fp32.engine --tag jetson_trt_fp32
```
This uses TensorRT through `cuda-python` (no PyTorch needed). **Reference (MAXN_SUPER, 3 objects per frame): total about 5.9 ms for FP16 and 6.2 ms for FP32, against 11.3 ms for ONNX Runtime on the CPU.**
If it fails with `ModuleNotFoundError: cuda`, run `pip3 install --user "cuda-python>=12.6,<13"`; if `import tensorrt` fails with `libnvdla_compiler.so` or `libcudla.so.1` missing, run
`sudo apt install nvidia-l4t-dla-compiler libcudla-12-6 && sudo ldconfig` (both are in `scripts/jetson_install.sh`). If it still fails, write down the exact error and carry on with the ONNX and `trtexec` numbers.
(ONNX Runtime from PyPI is CPU-only on the Jetson; the GPU path is this TensorRT backend.)

---

## 4. The full ROS 2 demo (simulation shown on the Jetson)
One command starts four nodes (simulated world, perception, planner, viewer window):
```bash
scripts/run_jetson_demo.sh
```
(Add `model_path:=$PWD/models/perception_fp16.engine` to use TensorRT. The simulated camera is slow on the Jetson CPU, so the window updates at only a few frames per second; that is normal.)

**What you should see** within ~15 seconds: a window called *Automator demo* with the **simulation on the left** (arm, trays, objects) and the **perception view on the right** (coloured boxes: green = recognised shape with its name and uncertainty `u`, red = UNKNOWN). Along the bottom, status lines: ARM IDLE/MOVING, last target, ESP32 status. The arm picks one object at a time and sorts it into its tray. When the table is empty it refills automatically.

Keyboard (click the window first): **e** = e-stop, **r** = reset, **q** = quit.

Optional: add `gui:=true` to also open PyBullet's own 3-D window, or `save_video:=$PWD/run1.mp4` to record the viewer.

**Checks to run while it is going** (new terminal, with the environment lines from Part 1):
```bash
ros2 topic list                       # expect /camera/image_raw /arm_target /joint_commands /arm_busy /estop ...
ros2 topic hz /joint_commands         # expect ~50 Hz
ros2 topic hz /arm_target             # expect ~5 Hz while the arm is idle, silence while it moves (this is by design)
ros2 topic echo /arm_target --once    # x y z class uncertainty  (class 0 sphere, 1 cube, 2 cylinder, 3 brick)
```
The vision node prints `target class ... (N ms, M objects)`: **write down the typical N**, which is the on-device perception time.

**Test A - manual e-stop:** press **e** in the window. The status bar turns red (`E-STOP ACTIVE`) and the arm freezes. Press **r**: it resumes. Pass = both happen.

**Test B - unknown object triggers the e-stop.** Stop the demo (Ctrl+C in the launch terminal), then:
```bash
scripts/run_jetson_demo.sh inject_unknown:=true
```
One object on the table is a random unseen mesh. **Expected:** the perception view shows it in red as `UNKNOWN`, and the arm stops with E-STOP ACTIVE, usually within a few seconds, without picking it up. Repeat 5 times, each with a different layout: `scripts/run_jetson_demo.sh inject_unknown:=true seed:=1`, then `seed:=2` ... `seed:=5` (stop with Ctrl+C between runs). **Record how many of the 5 runs stopped** (reference: about 83 % in simulation; a miss means the detector did not see the object).

**Known and normal:** an object touching or overlapping another in the camera view can look like one strange object and trigger an e-stop. About 1 in 100 normal objects also scores above the threshold. Note these, but they are not bugs to fix.

Stop everything with Ctrl+C in the launch terminal.

---

## 5. ESP32 over USB (micro-ROS)

### 5.1 Flash it (Arduino IDE, on your laptop or on the Jetson)
1. **Arduino IDE 2.x.** *File > Preferences > Additional boards manager URLs*: add `https://espressif.github.io/arduino-esp32/package_esp32_index.json`.
2. *Tools > Board > Boards Manager*: install **"esp32 by Espressif Systems" version 2.0.17** (**not 3.x**: the code uses the 2.x API).
3. Libraries: *Library Manager* > install **AccelStepper** (Mike McCauley). For **micro_ros_arduino**: download the **humble** zip from `https://github.com/micro-ROS/micro_ros_arduino/releases` (a file named like `micro_ros_arduino-2.0.7-humble.zip`) and use *Sketch > Include Library > Add .ZIP Library*.
4. Open `firmware/esp32_microros/esp32_actuator/esp32_actuator.ino` (the folder must stay together; it contains `pins.h` and `stepper_driver.*`).
5. *Tools*: Board **ESP32 Dev Module**; select the port. Click **Upload**. If it hangs on `Connecting......`, **hold the BOOT button** on the board until it starts writing.
6. **Success** = `Hard resetting via RTS pin...` and no red compile errors. **This sketch has never been compiled before.** If it does not compile, copy the **full red error text** into your notes and send it back; do not try to rewrite the code.
7. Close the Serial Monitor. It must stay closed: the serial port now carries micro-ROS, not text.

### 5.2 Connect it to the Jetson and start the agent
Plug the ESP32 into the Jetson with the USB cable, then:
```bash
ls /dev/ttyUSB* /dev/ttyACM*                 # expect one device, e.g. /dev/ttyUSB0
sudo usermod -aG dialout $USER               # first time only, then LOG OUT AND IN again
scripts/start_microros_agent.sh              # leave this running in its own terminal
```
**Expected:** the agent prints some lines and, after you **press the EN (reset) button on the ESP32 once**, lines such as `create_client` / `create_participant` / `create_subscriber`. If nothing appears after 10 s, press EN again.

In another terminal (environment lines from Part 1):
```bash
ros2 topic list | grep -E "esp32|joint_commands|estop"      # expect /esp32/status and the others
ros2 topic echo /esp32/status
```
`/esp32/status` is an integer sent 5 times a second. Meaning of the number (the demo window decodes it for you): `+1` e-stop latched, `+2` motors enabled, `+4` a command arrived in the last 200 ms, `+8` hardware e-stop pin pulled low; the part above 256 is **how many `/joint_commands` the ESP32 has received** (value ÷ 256, rounded down).

### 5.3 ESP32 tests (with the demo from Part 4 running as well)
Start `scripts/run_jetson_demo.sh` in a third terminal. The viewer's third line should now say **`ESP32 linked`**.

| # | Test | Do | Expected |
|---|---|---|---|
| E1 | Link | `ros2 topic echo /esp32/status` | numbers arriving ~5/s; with the demo running `cmds_received` in the viewer climbs. **Write down how fast** (compare with 50 Hz): note commands received in 10 s |
| E2 | Commands are fresh | demo running | viewer shows `cmd_fresh=yes`, `motors=ON` |
| E3 | Watchdog | stop only `joint_trajectory_node` (or Ctrl+C the whole demo) | within ~0.2 s `cmd_fresh=no` and `motors=off` |
| E4 | Software e-stop | press **e** in the viewer | `estop=YES`, the ESP32 board's blue LED (GPIO2) lights, `motors=off` |
| E5 | Reset | press **r** | `estop=no`, LED off |
| E6 | Unknown-object stop | `scripts/run_jetson_demo.sh inject_unknown:=true` | the ESP32 shows `estop=YES` when the arm stops |
| E7 | Hardware e-stop | touch a jumper wire between **GPIO33** and **GND** | immediately `estop=YES` and the `hw` flag; remove the wire, press **r** to clear |

If the ESP32 link drops (status stops), press EN on the ESP32 and restart the agent; note how often this happens.

---

## 6. Collect and send the results
Fill in this table (copy it into a message) and attach the listed files.

| Item | Your value |
|---|---|
| Jetson model, JetPack version (`cat /etc/nv_tegra_release`), power mode | |
| ONNX Runtime benchmark: total mean / p95 (ms) | |
| TensorRT FP16 / FP32 GPU compute mean (ms) from trtexec (reference 0.89 / 1.10) | |
| Full-pipeline TensorRT benchmark worked? (yes / no, error) | |
| Vision node time per frame in the ROS demo (typical ms) | |
| Sorting seen in the demo: roughly how many objects sorted right / wrong | |
| Test A, B (x/5 runs stopped), E1-E7: pass / fail, notes | |
| Commands received by the ESP32 in 10 s (should be ~500) | |
| Anything odd (crashes, window lag, ESP32 resets) | |

Attach: `jetson_selfcheck.txt`, `models/benchmark_jetson_*.json`, `trtexec_fp16.log`, `trtexec_fp32.log`, a screenshot or `run1.mp4` of the demo window, and, if you can, a tegrastats sample taken during the demo:
```bash
timeout 30 tegrastats --interval 500 | tee tegrastats_demo.log      # run while the demo is going
```

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ros2: command not found` | `source /opt/ros/humble/setup.bash` |
| `Package 'arm_perception' not found` | `source ros2_ws/install/setup.bash`; if still missing, rebuild (Part 1) |
| `Cannot find the Automator repo (perception/)` | `export AUTOMATOR_ROOT=$PWD` in the repo root, and build with `--symlink-install` |
| `ModuleNotFoundError: pybullet` / `onnxruntime` | `pip3 install --user "numpy<2" pybullet onnxruntime`  (keep `numpy<2`: NumPy 2 breaks the system OpenCV with `_ARRAY_API not found`); make sure you use `python3` (3.10), not a conda/venv Python |
| pybullet install takes ages or fails | normal to take 10-20 min; if it fails: `sudo apt install build-essential python3-dev cmake libgl1-mesa-dev` and retry |
| No window appears / `cannot connect to display` | run from a terminal **on the Jetson desktop**; over SSH use `export DISPLAY=:0` (desktop must be logged in) or run with `viewer:=false save_video:=$PWD/run1.mp4` |
| Window is black or "waiting for image..." | wait 15 s; check `ros2 topic hz /camera/image_raw` (should be ~3 Hz). If 0, `sim_bridge_node` crashed: read its error in the launch terminal |
| Camera rate far below 3 Hz | the CPU renders the simulated camera; run Part 2 power mode commands, and try `camera_hz:=2.0` |
| Arm never moves | perception only runs when the arm is idle for 0.6 s; check `/arm_target` is being published (`ros2 topic hz /arm_target`); a permanent e-stop shows red - press **r** |
| Immediate e-stop on every launch | an object may be touching another or the edge; press **r**. If it happens every time on a fresh start, report it |
| `trtexec: not found` | `export TRTEXEC=$(find /usr -name trtexec | head -1)` |
| `No module named cuda` / `libnvdla_compiler.so` or `libcudla.so.1` not found in the TensorRT benchmark | see 3b: install `cuda-python`, or the DLA packages and run `sudo ldconfig` |
| `/dev/ttyUSB0` permission denied | `sudo usermod -aG dialout $USER`, log out and in |
| No `/dev/ttyUSB*` after plugging in | charge-only cable, or board needs the CP210x/CH340 driver (normally built into JetPack); try another cable |
| Agent runs but `/esp32/status` never appears | press EN on the ESP32 **after** the agent started; close the Arduino Serial Monitor; check the agent was started on the right device |
| Upload fails `Failed to connect` | hold BOOT while it says `Connecting...`; try a lower upload speed in *Tools* |
| Arduino compile error | send back the full text; make sure board package is **2.0.17** and the **humble** micro_ros_arduino zip is installed |

## 8. Safety rules
- **No motors, drivers, motor power or arm** on the ESP32 during these tests.
- Keep the ESP32 on USB power only. Do not apply 12/24 V to anything.
- If something smells hot or behaves strangely, unplug the ESP32.
- Do not edit the code during testing. Write down what happened and send it back.
