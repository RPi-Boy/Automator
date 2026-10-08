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
    """Runs a serialized engine with torch CUDA tensors as device buffers (needs NVIDIA's CUDA-enabled PyTorch
    wheel on the Jetson, no pycuda). Supports the TensorRT 10 tensor API (JetPack 6.1+) and the older 8.x binding
    API. UNTESTED off-device - check `python3 -c "import tensorrt; print(tensorrt.__version__)"`, and fall back
    to the .onnx file (ONNX Runtime) if this misbehaves."""

    OUTPUTS = ("logits", "offset", "uncertainty")

    def __init__(self, path):
        import tensorrt as trt
        import torch
        self.trt, self.torch = trt, torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.v10 = hasattr(self.engine, "num_io_tensors")
        self.name = f"tensorrt {trt.__version__}"

    def __call__(self, crop, bbox):
        torch = self.torch
        inputs = {"crop": torch.from_numpy(crop).cuda().contiguous(),
                  "bbox_feat": torch.from_numpy(bbox).cuda().contiguous()}
        outputs = {}
        stream = torch.cuda.current_stream().cuda_stream
        if self.v10:
            for name, t in inputs.items():
                self.context.set_input_shape(name, tuple(t.shape))
                self.context.set_tensor_address(name, t.data_ptr())
            for name in self.OUTPUTS:
                outputs[name] = torch.empty(tuple(self.context.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
                self.context.set_tensor_address(name, outputs[name].data_ptr())
            self.context.execute_async_v3(stream)
        else:
            eng = self.engine
            names = [eng.get_binding_name(i) for i in range(eng.num_bindings)]
            for i, n in enumerate(names):
                if eng.binding_is_input(i):
                    self.context.set_binding_shape(i, tuple(inputs[n].shape))
            for i, n in enumerate(names):
                if not eng.binding_is_input(i):
                    outputs[n] = torch.empty(tuple(self.context.get_binding_shape(i)), dtype=torch.float32, device="cuda")
            ptrs = [inputs[n].data_ptr() if eng.binding_is_input(i) else outputs[n].data_ptr()
                    for i, n in enumerate(names)]
            self.context.execute_v2(ptrs)
        torch.cuda.synchronize()
        return tuple(outputs[n].cpu().numpy() for n in self.OUTPUTS)


def load_backend(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".engine":
        return TensorRTBackend(path)
    if ext == ".onnx":
        return OnnxBackend(path)
    if ext in (".pt", ".pth"):
        return TorchBackend(path)
    raise ValueError(f"unsupported model file: {path}")
