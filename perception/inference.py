"""Inference backends for the exported perception model: (crop, bbox_feat) -> (logits, offset, uncertainty).

  .engine  TensorRT (Jetson; build it ON the Jetson with scripts/build_trt_engine.sh)
  .onnx    ONNX Runtime (any machine; CUDA/TensorRT providers are used if installed)
  .pt      PyTorch state dict (development machine)
All backends take float32 numpy arrays: crop (B,3,128,128) in 0-1 and bbox_feat (B,4).
"""
import os

import numpy as np


class OnnxBackend:
    def __init__(self, path):
        import onnxruntime as ort
        preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(path, providers=providers)
        self.name = f"onnxruntime[{self.session.get_providers()[0]}]"

    def __call__(self, crop, bbox):
        logits, offset, unc = self.session.run(None, {"crop": crop, "bbox_feat": bbox})
        return logits, offset, unc


class TorchBackend:
    def __init__(self, path):
        import torch
        from perception.mobilenet_uncertainty import DeployWrapper, MobileNetUncertaintyBackbone
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MobileNetUncertaintyBackbone(num_classes=4, feature_dim=128)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model = DeployWrapper(model).to(self.device).eval()
        self.name = f"torch[{self.device}]"

    def __call__(self, crop, bbox):
        t = self.torch
        with t.no_grad():
            out = self.model(t.from_numpy(crop).to(self.device), t.from_numpy(bbox).to(self.device))
        return tuple(o.cpu().numpy() for o in out)


class TensorRTBackend:
    """Runs a serialized TensorRT 10 engine (JetPack 6) through the CUDA runtime from `cuda-python`.
    Needs only `tensorrt` (apt: python3-libnvinfer) and `pip3 install --user "cuda-python>=12.6,<13"`; no PyTorch.
    Device buffers are allocated once for up to MAX_BATCH crops."""

    OUTPUTS = (("logits", 4), ("offset", 3), ("uncertainty", 1))
    MAX_BATCH = 8

    def __init__(self, path):
        import tensorrt as trt
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart
        self.cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("TensorRT 10 API required (JetPack 6). Use the .onnx file on older versions.")
        self.context = self.engine.create_execution_context()
        self.name = f"tensorrt {trt.__version__} (cuda-python)"
        self.stream = self._ok(cudart.cudaStreamCreate())
        nb = self.MAX_BATCH
        self.d_crop = self._ok(cudart.cudaMalloc(nb * 3 * 128 * 128 * 4))
        self.d_bbox = self._ok(cudart.cudaMalloc(nb * 4 * 4))
        self.d_out = {n: self._ok(cudart.cudaMalloc(nb * w * 4)) for n, w in self.OUTPUTS}

    @staticmethod
    def _ok(result):
        err, *vals = result
        if int(err) != 0:
            raise RuntimeError(f"CUDA error {err}")
        return vals[0] if vals else None

    def __call__(self, crop, bbox):
        cr, B = self.cudart, crop.shape[0]
        if B > self.MAX_BATCH:
            raise ValueError(f"batch {B} > {self.MAX_BATCH}")
        crop = np.ascontiguousarray(crop, dtype=np.float32)
        bbox = np.ascontiguousarray(bbox, dtype=np.float32)
        h2d = cr.cudaMemcpyKind.cudaMemcpyHostToDevice
        d2h = cr.cudaMemcpyKind.cudaMemcpyDeviceToHost
        self._ok(cr.cudaMemcpyAsync(self.d_crop, crop.ctypes.data, crop.nbytes, h2d, self.stream))
        self._ok(cr.cudaMemcpyAsync(self.d_bbox, bbox.ctypes.data, bbox.nbytes, h2d, self.stream))
        self.context.set_input_shape("crop", tuple(crop.shape))
        self.context.set_input_shape("bbox_feat", tuple(bbox.shape))
        self.context.set_tensor_address("crop", int(self.d_crop))
        self.context.set_tensor_address("bbox_feat", int(self.d_bbox))
        for n, _ in self.OUTPUTS:
            self.context.set_tensor_address(n, int(self.d_out[n]))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        outs = []
        for n, w in self.OUTPUTS:
            host = np.empty((B, w), dtype=np.float32)
            self._ok(cr.cudaMemcpyAsync(host.ctypes.data, self.d_out[n], host.nbytes, d2h, self.stream))
            outs.append(host.reshape(B) if n == "uncertainty" else host)
        self._ok(cr.cudaStreamSynchronize(self.stream))
        return tuple(outs)


def load_backend(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".engine":
        return TensorRTBackend(path)
    if ext == ".onnx":
        return OnnxBackend(path)
    if ext in (".pt", ".pth"):
        return TorchBackend(path)
    raise ValueError(f"unsupported model file: {path}")
