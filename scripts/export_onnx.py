import argparse, os, yaml, torch
import sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "ultralytics")
for p in (ROOT, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

try:  # provide a minimal cv2 stub if OpenCV isn't available
    import cv2  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - environment without libGL/cv2
    class CV2Stub(types.ModuleType):
        __file__ = "cv2-stub"
        IMREAD_COLOR = 1
        IMREAD_GRAYSCALE = 0
        INTER_LINEAR = 1
        def __getattr__(self, name):
            return lambda *a, **k: None

    sys.modules['cv2'] = CV2Stub('cv2')

try:  # provide a minimal sklearn stub if scikit-learn isn't available
    import sklearn  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - avoid optional dependency
    import importlib.machinery as _machinery
    skmod = types.ModuleType('sklearn')
    metrics = types.ModuleType('metrics')
    def _confusion_matrix(*args, **kwargs):
        return None
    metrics.confusion_matrix = _confusion_matrix
    skmod.metrics = metrics
    skmod.__spec__ = _machinery.ModuleSpec('sklearn', None)
    metrics.__spec__ = _machinery.ModuleSpec('sklearn.metrics', None)
    sys.modules['sklearn'] = skmod
    sys.modules['sklearn.metrics'] = metrics

# PyTorch >=2.6 defaults to weights_only=True, which breaks loading older checkpoints
torch_load = torch.load
def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return torch_load(*args, **kwargs)
torch.load = _torch_load

try:  # add_safe_globals was introduced in later PyTorch versions
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - older torch without safety API
    def add_safe_globals(*_args, **_kwargs):
        return None

def export_ultralytics(pt_path, onnx_path, imgsz, dynamic, channels):
    from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, PoseModel,
                                      RTDETRDetectionModel, SegmentationModel)
    add_safe_globals([DetectionModel, SegmentationModel, PoseModel,
                      ClassificationModel, RTDETRDetectionModel])
    from ultralytics import YOLO  # import lazily to avoid cv2 dependency when unused
    model = YOLO(pt_path)
    model.model.yaml['ch'] = channels
    model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, opset=12, simplify=True)
    default = os.path.splitext(pt_path)[0] + ".onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    os.replace(default, onnx_path)
    print(f"[OK] Exported: {onnx_path}")

def fallback_torch_export(pt_path, onnx_path, imgsz, input_name, dynamic, channels):
    from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, PoseModel,
                                      RTDETRDetectionModel, SegmentationModel)
    from ultralytics.yolo.utils.tal import make_anchors
    from ultralytics import YOLO  # reuse loader without relying on Ultralytics export
    add_safe_globals([DetectionModel, SegmentationModel, PoseModel,
                      ClassificationModel, RTDETRDetectionModel])
    model = YOLO(pt_path)
    model.model.yaml['ch'] = channels
    mdl = model.model
    dummy = torch.zeros(1, channels, imgsz, imgsz)
    # some task-specific heads expect a Detect.forward reference named `detect`
    # which can be missing when loading weights standalone
    if hasattr(mdl, "model") and len(mdl.model):
        last = mdl.model[-1]
        if last.__class__.__name__ == "Pose":
            if not hasattr(last, "detect"):
                def _detect(self, x):
                    for i in range(self.nl):
                        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
                    if self.training:
                        return x
                    bs = x[0].shape[0]
                    x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
                    bbox, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
                    bbox = self.dfl(bbox)
                    y = torch.cat((bbox, cls.sigmoid()), 1)
                    return (y, x)
                last.detect = _detect
            # Recompute anchors for keypoint heads to avoid shape mismatches
            with torch.no_grad():
                mdl.train()
                feat_out = mdl(dummy)
                feats = feat_out[0] if isinstance(feat_out, (list, tuple)) else feat_out
                anchors, strides = make_anchors(feats, last.stride, 0.5)
                last.anchors = anchors.transpose(0, 1)
                last.strides = strides.transpose(0, 1)
                mdl.eval()
    dyn = {input_name: {0: "batch"}} if dynamic else None
    torch.onnx.export(
        mdl, dummy, onnx_path,
        input_names=[input_name], output_names=["output"],
        opset_version=12, do_constant_folding=True, dynamic_axes=dyn
    )
    print(f"[OK] Exported (fallback): {onnx_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--dynamic", action="store_true")
    args = ap.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    pt = cfg["pt_path"]; imgsz = int(cfg.get("imgsz", 640))
    # infer input channels from config mean/std lists if not explicitly set
    channels = cfg.get("channels")
    if channels is None:
        channels = len(cfg.get("mean", [])) or len(cfg.get("std", [])) or 3
    channels = int(channels)
    onnx_path = os.path.join("onnx", f"{cfg['model_name']}-fp32.onnx")
    if channels != 3:
        print(f"[INFO] channels={channels} not supported by Ultralytics export. Using fallback...")
        fallback_torch_export(pt, onnx_path, imgsz, cfg.get("input_name","images"), args.dynamic, channels)
    else:
        try:
            export_ultralytics(pt, onnx_path, imgsz, args.dynamic, channels)
        except Exception as e:
            print(f"[WARN] Ultralytics export failed: {e}\nTrying fallback torch export...")
            fallback_torch_export(pt, onnx_path, imgsz, cfg.get("input_name","images"), args.dynamic, channels)
