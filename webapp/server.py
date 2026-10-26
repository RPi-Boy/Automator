#!/usr/bin/env python3
"""Automator edge-test dashboard. Standard library only (nothing to pip install).

Run on the Jetson:   python3 webapp/server.py            then open  http://<jetson-ip>:8080/?token=<printed token>
It runs a fixed set of test jobs (latency benchmarks, TensorRT build/trtexec, sorting runs with video), streams
their log, latency chart and live frame to the browser, and lets you download results and videos.

Safety: only the jobs listed in JOBS can run (no arbitrary commands), every request needs the access token,
and file downloads are restricted to the runs/ directory and the models/ results. It still executes code on
the Jetson, so keep it on a trusted network.
"""
import argparse
import io
import json
import mimetypes
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RUNS = os.path.join(ROOT, "runs")
PY = sys.executable
TOKEN_FILE = os.path.expanduser("~/.automator_dashboard_token")
mimetypes.add_type("video/mp4", ".mp4")

# ----------------------------------------------------------------------------- machine info / telemetry

TELEMETRY = {"source": "none"}
_tel_lock = threading.Lock()


def _read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def parse_tegrastats(line):
    t = {"source": "tegrastats"}
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if m:
        t["ram_used_mb"], t["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"CPU \[(.*?)\]", line)
    if m:
        loads = [int(x.split("%")[0]) for x in m.group(1).split(",") if "%" in x and not x.startswith("off")]
        if loads:
            t["cpu_pct"] = sum(loads) / len(loads)
    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        t["gpu_pct"] = int(m.group(1))
    temps = {k: float(v) for k, v in re.findall(r"(\w+)@([\d.]+)C", line)}
    if temps:
        t["temp_c"] = max(temps.values())
        t["temps"] = temps
    m = re.search(r"VDD_IN (\d+)mW", line)
    if m:
        t["power_w"] = int(m.group(1)) / 1000.0
    return t


def telemetry_thread():
    """tegrastats on a Jetson; /proc fallback elsewhere (so the page also works on a PC)."""
    global TELEMETRY
    if shutil.which("tegrastats"):
        try:
            p = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, text=True)
            for line in p.stdout:
                with _tel_lock:
                    TELEMETRY = parse_tegrastats(line)
            return
        except OSError:
            pass
    prev = None
    while True:
        try:
            with open("/proc/stat") as f:
                a = [int(x) for x in f.readline().split()[1:]]
            tot, idle = sum(a), a[3] + a[4]
            cpu = 0.0 if prev is None else 100.0 * (1 - (idle - prev[1]) / max(1, tot - prev[0]))
            prev = (tot, idle)
            mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
            with _tel_lock:
                TELEMETRY = {"source": "proc", "cpu_pct": cpu, "ram_total_mb": mem["MemTotal"] // 1024,
                             "ram_used_mb": (mem["MemTotal"] - mem["MemAvailable"]) // 1024}
        except Exception:
            pass
        time.sleep(1)


_probe_cache = {}


def probe(module):
    if module not in _probe_cache:
        r = subprocess.run([PY, "-c", f"import {module}"], capture_output=True)
        _probe_cache[module] = r.returncode == 0
    return _probe_cache[module]


def machine_info():
    nv = _read("/etc/nv_tegra_release").splitlines()
    power = ""
    if shutil.which("nvpmodel"):
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        power = r.stdout.replace("NV Power Mode:", "").strip().replace("\n", " ")
    engines = {p: os.path.exists(os.path.join(ROOT, "models", f"perception_{p}.engine")) for p in ("fp16", "fp32")}
    backends = {
        "onnx": os.path.exists(os.path.join(ROOT, "models", "perception.onnx")) and probe("onnxruntime"),
        "trt_fp16": engines["fp16"] and probe("tensorrt"),
        "trt_fp32": engines["fp32"] and probe("tensorrt"),
    }
    return {"hostname": socket.gethostname(), "machine": platform.platform(), "arch": platform.machine(),
            "jetson": nv[0] if nv else "", "power_mode": power, "python": platform.python_version(),
            "backends": backends, "engines": engines,
            "trtexec": os.path.exists("/usr/src/tensorrt/bin/trtexec"), "ffmpeg": bool(shutil.which("ffmpeg")),
            "ros": os.path.exists("/opt/ros/humble/setup.bash")}


# ----------------------------------------------------------------------------- job catalogue

BACKEND_MODEL = {"onnx": "models/perception.onnx", "trt_fp16": "models/perception_fp16.engine",
                 "trt_fp32": "models/perception_fp32.engine"}
BACKEND_LABEL = {"onnx": "ONNX Runtime (CPU)", "trt_fp16": "TensorRT FP16 (GPU)", "trt_fp32": "TensorRT FP32 (GPU)"}
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def _int(params, key, default, lo, hi):
    try:
        v = int(float(params.get(key, default)))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _enum(params, key, default, choices):
    v = params.get(key, default)
    return v if v in choices else default


def _bench_cmd(backend, run_dir, extra):
    return [PY, "scripts/benchmark_edge.py", "--model", BACKEND_MODEL[backend], "--jsonl", os.path.join(run_dir, "metrics.jsonl"),
            "--live-dir", run_dir, "--no-write", "--tag", backend] + extra


def job_selfcheck(p, run_dir):
    cmd = [PY, "scripts/jetson_selfcheck.py"]
    if not os.path.exists("/opt/ros/humble/setup.bash"):   # e.g. a PC: missing ROS is a warning, not a failure
        cmd.append("--skip-ros")
    return [cmd]


def job_bench(p, run_dir):
    b = _enum(p, "backend", "onnx", BACKEND_MODEL)
    return [_bench_cmd(b, run_dir, ["--iters", str(_int(p, "iters", 300, 20, 5000))])]


def job_compare(p, run_dir):
    avail = [b for b, ok in machine_info()["backends"].items() if ok]
    return [_bench_cmd(b, run_dir, ["--iters", str(_int(p, "iters", 300, 20, 5000))]) for b in avail]


def job_live(p, run_dir):
    b = _enum(p, "backend", "onnx", BACKEND_MODEL)
    return [_bench_cmd(b, run_dir, ["--seconds", str(_int(p, "seconds", 30, 5, 900)), "--hz", str(_int(p, "hz", 20, 1, 100))])]


def job_trtexec(p, run_dir):
    prec, batch = _enum(p, "precision", "fp16", ("fp16", "fp32")), _int(p, "batch", 3, 1, 8)
    eng = os.path.join(ROOT, "models", f"perception_{prec}.engine")
    return [[TRTEXEC, f"--loadEngine={eng}", f"--shapes=crop:{batch}x3x128x128,bbox_feat:{batch}x4",
             "--iterations=300", "--warmUp=500"]]


def job_build_engine(p, run_dir):
    return [["bash", "scripts/build_trt_engine.sh", _enum(p, "precision", "fp16", ("fp16", "fp32"))]]


def job_sort(p, run_dir):
    b = _enum(p, "backend", "onnx", BACKEND_MODEL)
    cmd = [PY, "scripts/sort_demo.py", "--model", BACKEND_MODEL[b], "--trials", str(_int(p, "trials", 3, 1, 20)),
           "--objects", str(_int(p, "objects", 4, 1, 6)), "--unknown", str(_int(p, "unknown", 0, 0, 2)),
           "--seed", str(_int(p, "seed", 1000, 0, 100000)), "--video-dir", os.path.join(run_dir, "videos"),
           "--live-dir", run_dir, "--metrics-jsonl", os.path.join(run_dir, "metrics.jsonl"),
           "--out", os.path.relpath(os.path.join(run_dir, "sort_results.json"), ROOT)]
    return [cmd]


def job_ros(p, run_dir):
    script = ("source /opt/ros/humble/setup.bash; "
              "echo '== nodes'; timeout 6 ros2 node list; echo '== topics'; timeout 6 ros2 topic list; "
              "echo '== /joint_commands rate'; timeout 5 ros2 topic hz /joint_commands 2>&1 | tail -4; "
              "echo '== /arm_target rate'; timeout 5 ros2 topic hz /arm_target 2>&1 | tail -4; "
              "echo '== /esp32/status (one sample)'; timeout 6 ros2 topic echo --once /esp32/status 2>&1 | head -5; "
              "echo '== /estop'; timeout 4 ros2 topic echo --once /estop 2>&1 | head -3")
    return [["bash", "-c", script]]


JOBS = {
    "selfcheck": {"title": "Self-check", "desc": "Imports, ROS, models, display, serial port, a render + detect + inference smoke test.",
                  "build": job_selfcheck, "params": []},
    "bench": {"title": "Latency benchmark", "desc": "Detection + network + IK/planning per frame on real simulated frames. Live chart and summary.",
              "build": job_bench, "params": [("backend", "backend"), ("iters", "number", 300, 20, 5000, "iterations")]},
    "live": {"title": "Live latency test", "desc": "Runs the pipeline continuously at a fixed rate so you can watch latency over time (thermal throttling, jitter).",
             "build": job_live, "params": [("backend", "backend"), ("seconds", "number", 30, 5, 900, "seconds"), ("hz", "number", 20, 1, 100, "rate Hz")]},
    "compare": {"title": "Compare all backends", "desc": "Benchmarks every available backend one after another (CPU vs TensorRT FP32 vs FP16).",
                "build": job_compare, "params": [("iters", "number", 300, 20, 5000, "iterations each")]},
    "sort": {"title": "Sorting run with video", "desc": "The simulated arm sorts objects by shape; one video per trial, live view, per-cycle perception latency.",
             "build": job_sort, "params": [("backend", "backend"), ("trials", "number", 3, 1, 20, "trials"), ("objects", "number", 4, 1, 6, "objects"),
                                           ("unknown", "number", 0, 0, 2, "unseen objects"), ("seed", "number", 1000, 0, 100000, "seed")]},
    "trtexec": {"title": "TensorRT GPU latency (trtexec)", "desc": "Pure GPU compute time of the engine, no Python.",
                "build": job_trtexec, "params": [("precision", "select", "fp16", ["fp16", "fp32"], "precision"), ("batch", "number", 3, 1, 8, "batch")]},
    "build_engine": {"title": "Build TensorRT engine", "desc": "Compile the ONNX model into an engine on this Jetson (takes 1-4 minutes).",
                     "build": job_build_engine, "params": [("precision", "select", "fp16", ["fp16", "fp32"], "precision")]},
    "ros": {"title": "ROS / ESP32 snapshot", "desc": "Read-only look at ROS nodes, topics, command rates and the ESP32 status topic.",
            "build": job_ros, "params": []},
}


def catalogue():
    out = []
    for name, j in JOBS.items():
        out.append({"name": name, "title": j["title"], "desc": j["desc"], "params": [list(x) for x in j["params"]]})
    return out


# ----------------------------------------------------------------------------- run management

class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.current = None      # dict(meta) of the running job
        self.proc = None
        self.stop_flag = False
        os.makedirs(RUNS, exist_ok=True)

    def start(self, job, params):
        if job not in JOBS:
            raise ValueError("unknown job")
        with self.lock:
            if self.current is not None:
                raise RuntimeError("another test is already running")
            run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + job
            run_dir = os.path.join(RUNS, run_id)
            os.makedirs(run_dir, exist_ok=True)
            cmds = JOBS[job]["build"](params, run_dir)
            if not cmds:
                shutil.rmtree(run_dir, ignore_errors=True)
                raise RuntimeError("nothing to run (no backend available?)")
            meta = {"id": run_id, "job": job, "title": JOBS[job]["title"], "params": params, "status": "running",
                    "started": time.time(), "ended": None, "returncode": None, "summary": {}}
            self._save(meta)
            self.current, self.stop_flag = meta, False
        threading.Thread(target=self._run, args=(meta, cmds, run_dir), daemon=True).start()
        return meta

    @staticmethod
    def _save(meta):
        with open(os.path.join(RUNS, meta["id"], "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    def _run(self, meta, cmds, run_dir):
        env = dict(os.environ, AUTOMATOR_ROOT=ROOT, PYTHONUNBUFFERED="1")
        rc = 0
        with open(os.path.join(run_dir, "log.txt"), "ab", buffering=0) as log:
            for cmd in cmds:
                log.write(f"\n$ {' '.join(cmd)}\n".encode())
                try:
                    self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                    rc = self.proc.wait()
                except Exception as e:  # noqa: BLE001
                    log.write(f"failed to start: {e}\n".encode())
                    rc = 127
                if self.stop_flag or rc != 0:
                    break
            try:
                self._postprocess(meta, run_dir, log)
            except Exception as e:  # noqa: BLE001
                log.write(f"\npost-processing error: {e}\n".encode())
        meta["returncode"], meta["ended"] = rc, time.time()
        meta["status"] = "stopped" if self.stop_flag else ("done" if rc == 0 else "failed")
        self._save(meta)
        with self.lock:
            self.current, self.proc = None, None

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.stop_flag = True
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                except OSError:
                    pass
                return True
        return False

    # -- summaries and video transcoding -------------------------------------------------------------
    def _postprocess(self, meta, run_dir, log):
        job = meta["job"]
        lines = read_jsonl(os.path.join(run_dir, "metrics.jsonl"), 0, 10 ** 9)[0]
        if job in ("bench", "compare", "live"):
            runs = [l["summary"] for l in lines if l.get("kind") == "done"]
            meta["summary"] = {"runs": [{"backend": r["backend"], "model": r["model"], "iters": r["iters"],
                                         "detect_ms": r["detection"]["mean_ms"], "network_ms": r["network"]["mean_ms"],
                                         "plan_ms": r["ik_and_planning"]["mean_ms"], "total_mean_ms": r["total"]["mean_ms"],
                                         "total_p95_ms": r["total"]["p95_ms"], "total_max_ms": r["total"]["max_ms"]} for r in runs]}
        elif job == "trtexec":
            txt = open(os.path.join(run_dir, "log.txt"), errors="ignore").read()
            m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms.*?percentile\(95%\) = ([\d.]+)", txt)
            t = re.search(r"Throughput: ([\d.]+) qps", txt)
            if m:
                meta["summary"] = {"gpu_mean_ms": float(m.group(3)), "gpu_p95_ms": float(m.group(5)), "gpu_max_ms": float(m.group(2)),
                                   "throughput_qps": float(t.group(1)) if t else None}
        elif job == "sort":
            path = os.path.join(run_dir, "sort_results.json")
            if os.path.exists(path):
                r = json.load(open(path))
                meta["summary"] = {k: r.get(k) for k in ("trials", "known_objects_total", "sorted_correctly", "success_rate", "estop_trials")}
            vids = sorted(os.listdir(os.path.join(run_dir, "videos"))) if os.path.isdir(os.path.join(run_dir, "videos")) else []
            if vids and shutil.which("ffmpeg"):
                log.write(b"\n== transcoding videos to H.264 so browsers can play them\n")
                for v in vids:
                    src = os.path.join(run_dir, "videos", v)
                    tmp = src + ".h264.mp4"
                    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
                                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
                                       stdout=log, stderr=subprocess.STDOUT)
                    if r.returncode == 0:
                        os.replace(tmp, src)
            elif vids:
                log.write(b"\nffmpeg not installed: videos are MPEG-4 part 2 (download them; browsers cannot play them inline)\n")
        elif job == "selfcheck":
            p = os.path.join(ROOT, "jetson_selfcheck.txt")
            if os.path.exists(p):
                shutil.copy(p, os.path.join(run_dir, "jetson_selfcheck.txt"))
                txt = open(p).read()
                meta["summary"] = {"fails": txt.count("[FAIL]"), "warns": txt.count("[WARN]"), "passes": txt.count("[PASS]")}


def read_jsonl(path, offset=0, limit=2000):
    """Complete JSON lines starting at byte `offset`. Returns (lines, new_offset); a half-written last line is left for later."""
    lines, new_off = [], offset
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            for raw in f:
                if not raw.endswith(b"\n"):
                    break
                new_off += len(raw)
                try:
                    lines.append(json.loads(raw))
                except ValueError:
                    pass
                if len(lines) >= limit:
                    break
    except OSError:
        pass
    return lines, new_off


def list_runs():
    out = []
    if not os.path.isdir(RUNS):
        return out
    for name in sorted(os.listdir(RUNS), reverse=True):
        mp = os.path.join(RUNS, name, "meta.json")
        if os.path.exists(mp):
            try:
                meta = json.load(open(mp))
            except ValueError:
                continue
            meta["files"] = list_run_files(name)
            out.append(meta)
    return out


def list_run_files(run_id):
    base = os.path.join(RUNS, run_id)
    files = []
    for dirpath, _, names in os.walk(base):
        for n in sorted(names):
            if n in ("live.jpg", "live.tmp.jpg", "meta.json") or n.endswith(".h264.mp4"):
                continue
            full = os.path.join(dirpath, n)
            files.append({"path": os.path.relpath(full, base), "size": os.path.getsize(full)})
    return files


RUNNER = Runner()

# ----------------------------------------------------------------------------- HTTP

def load_token():
    t = os.environ.get("AUTOMATOR_TOKEN")
    if t:
        return t
    try:
        t = open(TOKEN_FILE).read().strip()
        if t:
            return t
    except OSError:
        pass
    t = secrets.token_urlsafe(12)
    with open(TOKEN_FILE, "w") as f:
        f.write(t)
    os.chmod(TOKEN_FILE, 0o600)
    return t


TOKEN = ""


class Handler(BaseHTTPRequestHandler):
    server_version = "AutomatorDashboard/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    # -- helpers
    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _auth(self, qs):
        cookie = self.headers.get("Cookie", "")
        got = (qs.get("token") or [None])[0] or self.headers.get("X-Token") or \
            (re.search(r"(?:^|; )tok=([^;]+)", cookie).group(1) if re.search(r"(?:^|; )tok=([^;]+)", cookie) else None)
        return got is not None and secrets.compare_digest(got, TOKEN)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 100_000:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _run_dir(self, run_id):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}_[a-z_]+", run_id or ""):
            return None
        d = os.path.join(RUNS, run_id)
        return d if os.path.isdir(d) else None

    # -- routes
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        path = u.path
        if not self._auth(qs):
            if path == "/":
                return self._send(401, "<h3>Access token required</h3><p>Open the URL printed by the server, "
                                       "which ends in <code>/?token=...</code></p>", "text/html")
            return self._send(401, {"error": "unauthorized"})
        try:
            if path == "/":
                html = open(os.path.join(HERE, "static", "index.html"), "rb").read()
                return self._send(200, html, "text/html; charset=utf-8",
                                  {"Set-Cookie": f"tok={TOKEN}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"})
            if path == "/api/info":
                with _tel_lock:
                    tel = dict(TELEMETRY)
                return self._send(200, {"info": machine_info(), "telemetry": tel, "jobs": catalogue(),
                                        "current": RUNNER.current})
            if path == "/api/telemetry":
                with _tel_lock:
                    return self._send(200, {"telemetry": dict(TELEMETRY), "current": RUNNER.current and RUNNER.current["id"]})
            if path == "/api/runs":
                return self._send(200, {"runs": list_runs(), "current": RUNNER.current and RUNNER.current["id"]})
            m = re.fullmatch(r"/api/run/([^/]+)/(log|metrics|live\.jpg|zip|meta)", path)
            if m:
                return self._run_route(m.group(1), m.group(2), qs)
            m = re.fullmatch(r"/files/([^/]+)/(.+)", path)
            if m:
                return self._file(m.group(1), urllib.parse.unquote(m.group(2)), qs)
            if path == "/files-models":
                names = [n for n in sorted(os.listdir(os.path.join(ROOT, "models"))) if n.endswith((".json", ".mp4"))]
                return self._send(200, {"files": [{"path": n, "size": os.path.getsize(os.path.join(ROOT, "models", n))} for n in names]})
            m = re.fullmatch(r"/models-file/([^/]+)", path)
            if m:
                return self._serve_file(os.path.join(ROOT, "models"), m.group(1), qs)
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if not self._auth(urllib.parse.parse_qs(u.query)):
            return self._send(401, {"error": "unauthorized"})
        try:
            if u.path == "/api/run":
                body = self._body()
                meta = RUNNER.start(body.get("job"), body.get("params") or {})
                return self._send(200, meta)
            if u.path == "/api/stop":
                return self._send(200, {"stopped": RUNNER.stop()})
            return self._send(404, {"error": "not found"})
        except RuntimeError as e:
            return self._send(409, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def do_DELETE(self):
        u = urllib.parse.urlparse(self.path)
        if not self._auth(urllib.parse.parse_qs(u.query)):
            return self._send(401, {"error": "unauthorized"})
        m = re.fullmatch(r"/api/run/([^/]+)", u.path)
        d = self._run_dir(m.group(1)) if m else None
        if not d:
            return self._send(404, {"error": "not found"})
        if RUNNER.current and RUNNER.current["id"] == m.group(1):
            return self._send(409, {"error": "run is still running"})
        shutil.rmtree(d, ignore_errors=True)
        return self._send(200, {"deleted": m.group(1)})

    def _run_route(self, run_id, what, qs):
        d = self._run_dir(run_id)
        if not d:
            return self._send(404, {"error": "no such run"})
        if what == "meta":
            return self._send(200, json.load(open(os.path.join(d, "meta.json"))))
        if what == "log":
            off = int((qs.get("offset") or ["0"])[0])
            with open(os.path.join(d, "log.txt"), "rb") as f:
                f.seek(off)
                chunk = f.read(200_000)
            status = json.load(open(os.path.join(d, "meta.json")))["status"]
            return self._send(200, {"text": chunk.decode(errors="replace"), "offset": off + len(chunk), "status": status})
        if what == "metrics":
            off = int((qs.get("offset") or ["0"])[0])
            lines, new_off = read_jsonl(os.path.join(d, "metrics.jsonl"), off)
            return self._send(200, {"lines": lines, "offset": new_off})
        if what == "live.jpg":
            p = os.path.join(d, "live.jpg")
            if not os.path.exists(p):
                return self._send(404, {"error": "no frame yet"})
            return self._send(200, open(p, "rb").read(), "image/jpeg")
        if what == "zip":
            tmp = os.path.join(RUNS, f".{run_id}.zip")
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
                for f in list_run_files(run_id):
                    z.write(os.path.join(d, f["path"]), os.path.join(run_id, f["path"]))
            try:
                return self._stream(tmp, f"{run_id}.zip", "application/zip", True, {})
            finally:
                os.remove(tmp)

    def _file(self, run_id, rel, qs):
        d = self._run_dir(run_id)
        if not d:
            return self._send(404, {"error": "no such run"})
        return self._serve_file(d, rel, qs)

    def _serve_file(self, base, rel, qs):
        full = os.path.realpath(os.path.join(base, rel))
        if not full.startswith(os.path.realpath(base) + os.sep) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if full.endswith(".log") or full.endswith(".txt"):
            ctype = "text/plain; charset=utf-8"
        return self._stream(full, os.path.basename(full), ctype, "dl" in qs, {})

    def _stream(self, full, name, ctype, as_download, extra):
        size = os.path.getsize(full)
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if as_download:
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.end_headers()
        with open(full, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def lan_addresses():
    addrs = set()
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
        addrs.update(a for a in out if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", a))
    except OSError:
        pass
    if not addrs:   # distros whose hostname has no -I (Arch, Alpine): ask iproute2, skipping docker and bridges
        try:
            for line in subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"], capture_output=True, text=True).stdout.splitlines():
                f = line.split()
                if len(f) > 3 and not f[1].startswith(("docker", "br-", "veth")):
                    addrs.add(f[3].split("/")[0])
        except OSError:
            pass
    return sorted(addrs) or ["127.0.0.1"]


def main():
    global TOKEN
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    TOKEN = load_token()
    threading.Thread(target=telemetry_thread, daemon=True).start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print("Automator dashboard running. Open one of:")
    for a in lan_addresses():
        print(f"   http://{a}:{args.port}/?token={TOKEN}")
    if not os.environ.get("AUTOMATOR_TOKEN"):
        print("(the token is stored in", TOKEN_FILE + "; it stays the same across restarts)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        RUNNER.stop()


if __name__ == "__main__":
    main()
