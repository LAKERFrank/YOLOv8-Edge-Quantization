#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify a TensorRT engine."""

import argparse
import os
import shutil
import subprocess


def print_trtexec_info(engine_path: str):
    """Invoke trtexec (if available) to dump layer precisions."""
    trtexec = shutil.which("trtexec")
    if not trtexec:
        print("[WARN] trtexec command not found; ensure TensorRT is installed and in PATH.")
        return

    print(f"[INFO] trtexec: {trtexec}")

    cmd = [
        trtexec,
        f"--loadEngine={engine_path}",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        print("==== trtexec summary (truncated) ====")
        lines = [l for l in proc.stdout.splitlines() if "Layer" in l or "Precision" in l or "Binding" in l]
        print("\n".join(lines[:300]))
        print("=====================================")
    except subprocess.CalledProcessError as e:
        last = e.stdout.strip().splitlines()[-1] if e.stdout else ""
        print(f"[WARN] trtexec ({trtexec}) failed (returncode {e.returncode}): {last}")


def binding_summary_with_tensorrt(engine_path: str):
    try:
        import tensorrt as trt
    except Exception as e:
        print(f"[WARN] TensorRT python package not available: {e}")
        return
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        try:
            engine = rt.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"[WARN] Failed to deserialize engine with TensorRT {trt.__version__}: {e}")
            print("[WARN] Ensure the runtime TensorRT version is at least as new as the one used to build the engine.")
            return

    try:
        # Legacy APIs (TensorRT 8/9)
        nb = engine.num_bindings
        print(f"[INFO] num_bindings: {nb}")
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            print(f"  - {'INPUT ' if is_input else 'OUTPUT'} {i}: {name:30s} shape={tuple(shape)} dtype={dtype}")
    except AttributeError:
        # TensorRT 10 APIs
        trt_major = int(trt.__version__.split('.')[0])
        if trt_major >= 10 and hasattr(engine, "num_io_tensors"):
            nb = engine.num_io_tensors
            print(f"[INFO] num_io_tensors: {nb}")
            for i in range(nb):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)
                io = "INPUT " if mode == trt.TensorIOMode.INPUT else "OUTPUT"
                print(f"  - {io} {i}: {name:30s} shape={tuple(shape)} dtype={dtype}")
        else:
            print("[WARN] Ensure the runtime TensorRT version is at least as new as the one used to build the engine.")


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
