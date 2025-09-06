#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify a TensorRT engine:
- print input binding shapes (expect C=1)
- (optional) call trtexec to dump layer precisions
"""
import argparse, os, subprocess, sys

def print_trtexec_info(engine_path: str):
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        print("==== trtexec summary (truncated) ====")
        lines = [l for l in out.splitlines() if "Layer" in l or "Precision" in l or "Binding" in l]
        print("\n".join(lines[:300]))
        print("=====================================")
    except Exception as e:
        print(f"[WARN] trtexec not available or failed: {e}")

def binding_summary_with_tensorrt(engine_path: str):
    try:
        import tensorrt as trt
    except Exception as e:
        print(f"[WARN] TensorRT python package not available: {e}")
        return
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    try:
        # TRT 8:
        nb = engine.num_bindings
        print(f"[INFO] num_bindings: {nb}")
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            print(f"  - {'INPUT ' if is_input else 'OUTPUT'} {i}: {name:30s} shape={tuple(shape)} dtype={dtype}")
    except AttributeError:
        # TRT 9 new APIs (if needed, extend here)
        print("[WARN] Please extend for newer TensorRT API versions.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()

    if not os.path.isfile(args.engine):
        raise SystemExit(f"Engine not found: {args.engine}")

    print("[STEP] TensorRT API binding summary")
    binding_summary_with_tensorrt(args.engine)

    print("\n[STEP] trtexec layer info (if available)")
    print_trtexec_info(args.engine)

    print("\n[CHECK] Manually verify input channel C=1 in the binding shape (second dimension).")

if __name__ == "__main__":
    main()
