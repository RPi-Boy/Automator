"""Checks that this machine can run the Automator demo and writes jetson_selfcheck.txt (send that file back).

  python3 scripts/jetson_selfcheck.py            # run from the repository root
Prints PASS / WARN / FAIL per item. FAIL = the demo will not work; WARN = works but degraded.
"""
import glob
import importlib
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "arm_perception"))
LINES, FAILS = [], 0


def report(level, name, detail=""):
    global FAILS
    FAILS += level == "FAIL"
    line = f"[{level:4s}] {name}" + (f": {detail}" if detail else "")
    print(line)
    LINES.append(line)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def check_import(mod, need=True, hint=""):
    try:
        m = importlib.import_module(mod)
        report("PASS", f"import {mod}", getattr(m, "__version__", ""))
        return m
    except Exception as e:
        report("FAIL" if need else "WARN", f"import {mod}", f"{type(e).__name__}: {e}. {hint}")
        return None


def main():
    report("PASS", "machine", f"{platform.platform()} / {platform.machine()}")
    tegra = sh("cat /etc/nv_tegra_release | head -1")
    report("PASS" if tegra else "WARN", "Jetson L4T release", tegra or "not a Jetson (fine for a PC dry run)")
    mode = sh("nvpmodel -q 2>/dev/null | tr '\\n' ' '")
    if mode:
        report("PASS", "power mode", mode + "  (for benchmarks run: sudo nvpmodel -m 0 && sudo jetson_clocks)")
    v = sys.version_info
    report("PASS" if (v.major, v.minor) == (3, 10) else "WARN", "python", f"{v.major}.{v.minor}.{v.micro} (ROS 2 Humble expects 3.10)")

    np_ = check_import("numpy", hint="sudo apt install python3-numpy")
    check_import("cv2", hint="sudo apt install python3-opencv")
    pb = check_import("pybullet", hint="pip3 install pybullet (compiles on ARM, ~15 min)")
    ort = check_import("onnxruntime", hint="pip3 install onnxruntime")
    if ort:
        report("PASS", "onnxruntime providers", ", ".join(ort.get_available_providers()))
    check_import("tensorrt", need=False, hint="comes with JetPack; needed only for .engine files")
    torch = check_import("torch", need=False, hint="only needed for the TensorRT backend / .pt model")
    if torch:
        report("PASS" if torch.cuda.is_available() else "WARN", "torch CUDA", str(torch.cuda.is_available()))

    # ROS
    ros = shutil.which("ros2")
    report("PASS" if ros else "FAIL", "ros2 on PATH", ros or "run: source /opt/ros/humble/setup.bash")
    if ros:
        pkg = sh("ros2 pkg prefix arm_perception 2>/dev/null")
        report("PASS" if pkg else "FAIL", "arm_perception built", pkg or "run colcon build, then source ros2_ws/install/setup.bash")
        agent = sh("ros2 pkg prefix micro_ros_agent 2>/dev/null")
        report("PASS" if agent else "WARN", "micro_ros_agent installed", agent or "sudo apt install ros-humble-micro-ros-agent (or use docker, see instructions.md)")
    report("PASS" if os.environ.get("AUTOMATOR_ROOT") == ROOT else "WARN", "AUTOMATOR_ROOT", os.environ.get("AUTOMATOR_ROOT", f"unset - run: export AUTOMATOR_ROOT={ROOT}"))

    # files
    for f in ("models/perception.onnx", "models/perception.pt"):
        report("PASS" if os.path.exists(os.path.join(ROOT, f)) else "FAIL", f)
    engines = glob.glob(os.path.join(ROOT, "models", "*.engine"))
    report("PASS" if engines else "WARN", "TensorRT engine", ", ".join(os.path.basename(e) for e in engines) or "none yet (build it, see instructions.md)")

    # display + serial
    report("PASS" if os.environ.get("DISPLAY") else "WARN", "DISPLAY", os.environ.get("DISPLAY", "unset: no window possible; use save_video or run from the Jetson desktop / ssh -X"))
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    report("PASS" if ports else "WARN", "ESP32 serial port", ", ".join(ports) or "none (plug the ESP32 in with a data USB cable)")
    if ports:
        ok = os.access(ports[0], os.R_OK | os.W_OK)
        report("PASS" if ok else "FAIL", f"permission on {ports[0]}", "ok" if ok else "run: sudo usermod -aG dialout $USER, then log out and in")

    # functional: render + detect + infer
    if np_ and pb:
        try:
            import numpy
            import pybullet
            from perception.detector import detect
            from sim.scene import SortScene
            cid = pybullet.connect(pybullet.DIRECT)
            scene = SortScene(cid)
            scene.build_static()
            t = time.perf_counter()
            ref, _ = scene.camera.render(cid)
            report("PASS", "PyBullet render 640x480", f"{(time.perf_counter() - t) * 1000:.0f} ms/frame (3 Hz camera needs < 300)")
            scene.spawn([0, 1, 2], numpy.random.default_rng(0))
            rgb, _ = scene.camera.render(cid)
            dets = detect(rgb, ref, scene.camera)
            report("PASS" if len(dets) == 3 else "FAIL", "detector finds 3 spawned objects", f"found {len(dets)}")
            if ort and dets:
                from perception.inference import OnnxBackend
                be = OnnxBackend(os.path.join(ROOT, "models", "perception.onnx"))
                crops = numpy.ascontiguousarray(numpy.stack([d["crop"] for d in dets]).transpose(0, 3, 1, 2).astype("float32") / 255.0)
                bb = numpy.stack([d["bbox_feat"] for d in dets]).astype("float32")
                be(crops, bb)
                t = time.perf_counter()
                for _ in range(20):
                    logits, off, unc = be(crops, bb)
                got = sorted(logits.argmax(1).tolist())
                report("PASS" if got == [0, 1, 2] else "FAIL", f"inference via {be.name}",
                       f"{(time.perf_counter() - t) * 50:.1f} ms for 3 crops; classes found {got}, expected [0, 1, 2]")
        except Exception as e:
            report("FAIL", "functional test", f"{type(e).__name__}: {e}")

    out = os.path.join(ROOT, "jetson_selfcheck.txt")
    with open(out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\n{FAILS} FAIL. Wrote {out}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
