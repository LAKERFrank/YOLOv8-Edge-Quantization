#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export TensorRT engine from a YOLO(.pt) model with INT8 PTQ.
- Requires Ultralytics + TensorRT-capable environment.
"""
import argparse, os, shutil, sys
from ultralytics import YOLO

def parse_shape(s: str):
    # "1,1,640,640" -> (1,1,640,640)
    return tuple(int(x) for x in s.split(","))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to YOLO .pt (1ch)")
    ap.add_argument("--data", default="trt_quant/calib/calib.yaml", help="data yaml for INT8 calibration")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default=0)
    ap.add_argument("--int8", action="store_true", help="build INT8 engine")
    ap.add_argument("--fp16", action="store_true", help="build FP16 engine (optional)")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--minshape", default="1,1,480,640")
    ap.add_argument("--optshape", default="1,1,640,640")
    ap.add_argument("--maxshape", default="1,1,1080,1920")
    ap.add_argument("--outdir", default="trt_quant/engine")
    ap.add_argument("--name", default=None, help="output engine name (auto if None)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    export_kwargs = dict(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
    if args.dynamic:
        export_kwargs["dynamic"] = True
        # TensorRT workspace size in GB (ignored if unsupported).
        try:
            export_kwargs["workspace"] = 2
        except Exception:
            pass
        # Pack optimization profiles (min,opt,max). Some Ultralytics versions
        # do not accept the "shape" argument, so export will fallback below.
        export_kwargs["shape"] = (
            parse_shape(args.minshape),
            parse_shape(args.optshape),
            parse_shape(args.maxshape),
        )

    if args.int8:
        export_kwargs["int8"] = True
        export_kwargs["data"] = args.data
        tag = "int8"
    elif args.fp16:
        export_kwargs["half"] = True
        tag = "fp16"
    else:
        tag = "fp32"

    print("[INFO] Exporting TensorRT engine with args:", export_kwargs)
    try:
        engine_path = model.export(**export_kwargs)  # returns path to .engine
    except SyntaxError as e:
        if "shape" in str(e):
            print("[WARN] 'shape' arg unsupported by this Ultralytics version; retrying without it")
            export_kwargs.pop("shape", None)
            engine_path = model.export(**export_kwargs)
        else:
            raise

    # Move/rename into outdir
    out_name = args.name or f"pose_{tag}.engine"
    out_path = os.path.join(args.outdir, out_name)
    shutil.move(engine_path, out_path)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
