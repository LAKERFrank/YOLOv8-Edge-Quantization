import argparse, os, sys, types, yaml, torch

# Ensure the repository's custom ultralytics package (which contains TrackNet) is
# importable even when a global ultralytics installation exists.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ULTRA_DIR = os.path.join(REPO_DIR, "ultralytics")
if ULTRA_DIR not in sys.path:
    sys.path.insert(0, ULTRA_DIR)

# Ultralytics unconditionally imports OpenCV.  In minimal environments this may
# fail due to missing system libraries (e.g. libGL).  Stub a minimal module so
# that model loading can proceed without a full OpenCV installation.
try:  # pragma: no cover - best effort
    import cv2  # noqa: F401
except Exception:  # pragma: no cover - headless environments
    class _CV2Stub:
        IMREAD_COLOR = 1
        IMREAD_GRAYSCALE = 0
        def __getattr__(self, _):
            return lambda *a, **k: None
        setNumThreads = staticmethod(lambda *a, **k: None)
        ocl = types.SimpleNamespace(setUseOpenCL=lambda *a, **k: None)
    sys.modules["cv2"] = _CV2Stub()

# Some TrackNet utilities require scikit-learn for metrics, but importing the
# real package is unnecessary for ONNX export. Provide a tiny stub instead.
try:  # pragma: no cover
    import sklearn  # noqa: F401
except Exception:  # pragma: no cover - minimal environments
    metrics_stub = types.ModuleType("sklearn.metrics")
    metrics_stub.confusion_matrix = lambda *a, **k: None
    sklearn_stub = types.ModuleType("sklearn")
    sklearn_stub.metrics = metrics_stub
    sys.modules.update({
        "sklearn": sklearn_stub,
        "sklearn.metrics": metrics_stub,
    })

# TorchVision is also imported during model loading; provide a very small stub if
# it is unavailable (common in minimal CPU-only environments).
try:  # pragma: no cover
    import torchvision  # noqa: F401
except Exception:  # pragma: no cover - minimal environments
    tv_stub = types.ModuleType("torchvision")
    tv_stub.__version__ = "0.0"
    tv_stub.ops = types.SimpleNamespace(nms=lambda *a, **k: torch.zeros((0,), dtype=torch.int64))
    tv_stub.datasets = types.SimpleNamespace(ImageFolder=object)
    tv_stub.transforms = types.SimpleNamespace()
    sys.modules["torchvision"] = tv_stub

from ultralytics import YOLO

def export_ultralytics(pt_path, onnx_path, imgsz, dynamic):
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, opset=12, simplify=True)
    default = os.path.splitext(pt_path)[0] + ".onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    os.replace(default, onnx_path)
    print(f"[OK] Exported: {onnx_path}")

def fallback_torch_export(pt_path, onnx_path, imgsz, input_name, dynamic):
    """Fallback ONNX export using torch.onnx.export directly."""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    mdl = ckpt["model"].eval() if isinstance(ckpt, dict) else ckpt.eval()
    mdl = mdl.float()
    first = mdl.model[0]
    in_ch = getattr(getattr(first, "conv", first), "in_channels")
    dummy = torch.zeros(1, in_ch, imgsz, imgsz, dtype=next(mdl.parameters()).dtype)
    dyn = {input_name: {0: "batch"}} if dynamic else None
    torch.onnx.export(
        mdl, dummy, onnx_path,
        input_names=[input_name], output_names=["output"],
        opset_version=12, do_constant_folding=True, dynamic_axes=dyn,
    )
    print(f"[OK] Exported (fallback): {onnx_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--dynamic", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    pt = cfg["pt_path"]; imgsz = int(cfg.get("imgsz", 640))
    onnx_path = os.path.join("onnx", f"{cfg['model_name']}-fp32.onnx")
    try:
        export_ultralytics(pt, onnx_path, imgsz, args.dynamic)
    except Exception as e:
        print(f"[WARN] Ultralytics export failed: {e}\nTrying fallback torch export...")
        fallback_torch_export(pt, onnx_path, imgsz, cfg.get("input_name","images"), args.dynamic)
