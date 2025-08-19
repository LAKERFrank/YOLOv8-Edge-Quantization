import argparse, os, yaml, torch
import sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "ultralytics")
for p in (ROOT, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

try:  # provide a minimal cv2 stub if OpenCV isn't available
    import cv2  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - environment without libGL/cv2
    class CV2Stub(types.ModuleType):
        __file__ = "cv2-stub"
        IMREAD_COLOR = 1
        IMREAD_GRAYSCALE = 0
        INTER_LINEAR = 1
        def __getattr__(self, name):
            return lambda *a, **k: None

    sys.modules['cv2'] = CV2Stub('cv2')

def export_ultralytics(pt_path, onnx_path, imgsz, dynamic, channels):
    from ultralytics import YOLO  # import lazily to avoid cv2 dependency when unused
    model = YOLO(pt_path)
    model.model.yaml['ch'] = channels
    model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, opset=12, simplify=True)
    default = os.path.splitext(pt_path)[0] + ".onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    os.replace(default, onnx_path)
    print(f"[OK] Exported: {onnx_path}")

def fallback_torch_export(pt_path, onnx_path, imgsz, input_name, dynamic, channels):
    from ultralytics import YOLO  # reuse loader without relying on Ultralytics export
    model = YOLO(pt_path)
    model.model.yaml['ch'] = channels
    mdl = model.model
    mdl.eval()
    dummy = torch.zeros(1, channels, imgsz, imgsz)
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
    cfg = yaml.safe_load(open(args.cfg))
    pt = cfg["pt_path"]; imgsz = int(cfg.get("imgsz", 640))
    channels = int(cfg.get("channels", 3))
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
