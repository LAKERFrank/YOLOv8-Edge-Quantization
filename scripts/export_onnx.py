import argparse, os, yaml, torch
from ultralytics import YOLO

def export_ultralytics(pt_path, onnx_path, imgsz, dynamic):
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, opset=12, simplify=True)
    default = os.path.splitext(pt_path)[0] + ".onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    os.replace(default, onnx_path)
    print(f"[OK] Exported: {onnx_path}")

def fallback_torch_export(pt_path, onnx_path, imgsz, input_name, dynamic):
    mdl = torch.load(pt_path, map_location="cpu")
    mdl.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
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
    onnx_path = os.path.join("onnx", f"{cfg['model_name']}-fp32.onnx")
    try:
        export_ultralytics(pt, onnx_path, imgsz, args.dynamic)
    except Exception as e:
        print(f"[WARN] Ultralytics export failed: {e}\nTrying fallback torch export...")
        fallback_torch_export(pt, onnx_path, imgsz, cfg.get("input_name","images"), args.dynamic)
