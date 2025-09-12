#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a TensorRT engine from a YOLO (.pt) model using trtexec.

Flow: .pt -> .onnx via Ultralytics -> trtexec build (INT8/FP16/FP32).
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_shape(shape: str):
    """"1,1,640,640" -> (1, 1, 640, 640)."""
    return tuple(int(x) for x in shape.split(","))


def shape_to_trtexec(shape: str) -> str:
    """Convert comma-separated dims to '1x1x640x640' for trtexec."""
    return "x".join(shape.split(","))


def main() -> None:
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
    ap.add_argument("--workspace", type=int, default=2048, help="builder workspace (MB)")
    ap.add_argument("--outdir", default="trt_quant/engine")
    ap.add_argument("--name", default=None, help="output engine name (auto if None)")
    args = ap.parse_args()

    onnx_kwargs = dict(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    if args.dynamic:
        onnx_kwargs["dynamic"] = True
        onnx_kwargs["shape"] = (
            parse_shape(args.minshape),
            parse_shape(args.optshape),
            parse_shape(args.maxshape),
        )

    if args.int8:
        onnx_kwargs["int8"] = True
        onnx_kwargs["data"] = args.data
        tag = "int8"
    elif args.fp16:
        onnx_kwargs["half"] = True
        tag = "fp16"
    else:
        tag = "fp32"

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - missing dependency
        print(f"[ERROR] Ultralytics not available: {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print("[INFO] Exporting ONNX with args:", onnx_kwargs)
    try:
        onnx_path = Path(model.export(**onnx_kwargs))
    except SyntaxError as e:
        if "shape" in str(e):
            print("[WARN] 'shape' arg unsupported by this Ultralytics version; retrying without it")
            onnx_kwargs.pop("shape", None)
            print("[INFO] Exporting ONNX with args:", onnx_kwargs)
            onnx_path = Path(model.export(**onnx_kwargs))
        else:
            raise

    engine_name = args.name or f"pose_{tag}.engine"
    engine_path = Path(args.outdir) / engine_name

    trtexec_bin = shutil.which("trtexec")
    if not trtexec_bin:
        print("[ERROR] trtexec not found on PATH", file=sys.stderr)
        sys.exit(1)

    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--explicitBatch",
        f"--workspace={args.workspace}",
    ]
    if args.dynamic:
        input_name = "images"
        cmd += [
            f"--minShapes={input_name}:{shape_to_trtexec(args.minshape)}",
            f"--optShapes={input_name}:{shape_to_trtexec(args.optshape)}",
            f"--maxShapes={input_name}:{shape_to_trtexec(args.maxshape)}",
        ]
    else:
        cmd.append(f"--optShapes=images:{shape_to_trtexec(args.optshape)}")

    if args.int8:
        cmd.append("--int8")
    elif args.fp16:
        cmd.append("--fp16")

    print("[INFO] Running:", " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(f"[ERROR] trtexec failed (returncode {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[OK] Saved: {engine_path}")


if __name__ == "__main__":
    main()
