#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export TensorRT engine from a YOLO(.pt) model.
- Flow: .pt -> .onnx -> trtexec build (INT8/FP16/FP32)
- Requires Ultralytics + TensorRT-capable environment with `trtexec` available.
"""
import argparse, os, shutil, subprocess, sys
from ultralytics import YOLO


def to_trt_shape(s: str) -> str:
    # "1,1,640,640" -> "1x1x640x640"
    return "x".join(s.split(","))

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
    ap.add_argument("--trtexec", default="trtexec", help="path to trtexec binary")
    ap.add_argument("--calib", default=None, help="calibration cache/data path for INT8")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    onnx_kwargs = dict(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    if args.dynamic:
        onnx_kwargs["dynamic"] = True

    print("[INFO] Exporting ONNX with args:", onnx_kwargs)
    onnx_path = model.export(**onnx_kwargs)  # returns path to .onnx

    tmp_engine = os.path.join(args.outdir, "tmp.engine")
    cmd = [
        args.trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={tmp_engine}",
    ]

    if args.dynamic:
        cmd.extend(
            [
                f"--minShapes=images:{to_trt_shape(args.minshape)}",
                f"--optShapes=images:{to_trt_shape(args.optshape)}",
                f"--maxShapes=images:{to_trt_shape(args.maxshape)}",
            ]
        )
    else:
        cmd.append(f"--shapes=images:{to_trt_shape(args.optshape)}")

    if args.int8:
        cmd.append("--int8")
        if args.calib:
            cmd.append(f"--calib={args.calib}")
        tag = "int8"
    elif args.fp16:
        cmd.append("--fp16")
        tag = "fp16"
    else:
        tag = "fp32"

    print("[INFO] Running trtexec:", " ".join(cmd))
    subprocess.check_call(cmd)

    out_name = args.name or f"pose_{tag}.engine"
    out_path = os.path.join(args.outdir, out_name)
    shutil.move(tmp_engine, out_path)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
