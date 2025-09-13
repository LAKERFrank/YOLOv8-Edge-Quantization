#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference with a TensorRT engine using Ultralytics runtime.
Works for YOLO-Pose engine produced by export_trt.py.
"""
import argparse, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to .engine")
    ap.add_argument("--source", required=True, help="image/video path or dir")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--save", action="store_true", help="save rendered outputs")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--task", default="pose", help="model task type (detect, pose, etc.)")
    ap.add_argument("--channels", type=int, default=1, help="input channels of the engine")
    args = ap.parse_args()

    # Explicitly pass task to avoid incorrect automatic guessing
    model = YOLO(args.engine, task=args.task)
    # Ensure channel count matches engine expectation when metadata is absent
    if hasattr(model, "predictor") and hasattr(model.predictor, "overrides"):
        model.predictor.overrides["channels"] = args.channels

    results = model.predict(source=args.source, imgsz=args.imgsz, device=args.device,
                            conf=args.conf, save=args.save, stream=True, verbose=False)

    n_imgs = 0
    for r in results:
        n_imgs += 1
        n_boxes = 0 if r.boxes is None else len(r.boxes)
        n_kpts  = 0 if getattr(r, "keypoints", None) is None else len(r.keypoints)
        print(f"[{n_imgs}] boxes={n_boxes}, keypoints={n_kpts}")
    print(f"[OK] Done. processed frames/images: {n_imgs}")

if __name__ == "__main__":
    main()
