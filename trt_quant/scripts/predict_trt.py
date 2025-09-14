#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference with a TensorRT INT8 engine for pose estimation.
- Supports image or video input.
- Saves annotated output and prints bounding boxes and keypoints.
"""
import argparse
import os

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to TensorRT engine")
    ap.add_argument("--source", required=True, help="image or video path")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--device", default=0, help="CUDA device")
    ap.add_argument("--show", action="store_true", help="display results in a window")
    args = ap.parse_args()

    if not os.path.isfile(args.engine):
        raise SystemExit(f"Engine not found: {args.engine}")

    model = YOLO(args.engine)

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        task="pose",
        save=True,
        show=args.show,
        stream=True,
    )

    for r in results:
        boxes = r.boxes
        kpts = getattr(r, "keypoints", None)
        if boxes is not None and len(boxes) > 0:
            print("Boxes (xyxy):", boxes.xyxy.cpu().numpy().tolist())
        if kpts is not None and len(kpts) > 0:
            print("Keypoints (xy):", kpts.xy.cpu().numpy().tolist())


if __name__ == "__main__":
    main()
