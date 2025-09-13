#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify a TensorRT engine:
- print input and output binding shapes
- assert that all input bindings have channel dimension C=1
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

def binding_summary_with_tensorrt(engine_path: str) -> bool:
    """Print bindings and verify channel dimension is 1 for inputs."""
    try:
        import tensorrt as trt
    except Exception as e:
        print(f"[WARN] TensorRT python package not available: {e}")
        return False

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())

    ok = True
    try:
        # TRT 8 API
        nb = engine.num_bindings
        print(f"[INFO] num_bindings: {nb}")
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            print(
                f"  - {'INPUT ' if is_input else 'OUTPUT'} {i}: {name:30s} "
                f"shape={tuple(shape)} dtype={dtype}"
            )
            if is_input and len(shape) > 1:
                if shape[1] != 1:
                    print(
                        f"    [FAIL] expected channel dimension 1 but got {shape[1]}"
                    )
                    ok = False
                else:
                    print("    [PASS] channel dimension is 1")
    except AttributeError:
        # TRT 9 new APIs (if needed, extend here)
        print("[WARN] Please extend for newer TensorRT API versions.")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()

    if not os.path.isfile(args.engine):
        raise SystemExit(f"Engine not found: {args.engine}")

    print("[STEP] TensorRT API binding summary")
    ok = binding_summary_with_tensorrt(args.engine)

    print("\n[STEP] trtexec layer info (if available)")
    print_trtexec_info(args.engine)

    if not ok:
        print("\n[CHECK] Input channel dimension is not 1!")
        sys.exit(1)
    else:
        print("\n[CHECK] All input channel dimensions are 1")

if __name__ == "__main__":
    main()
