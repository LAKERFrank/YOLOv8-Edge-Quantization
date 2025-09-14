#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run pose inference with a TensorRT engine.

This utility works with INT8 engines exported by ``export_trt.py`` and
supports image, directory, or video inputs. If the engine expects a
single-channel image (C=1), sources are converted to grayscale
automatically.

Bounding boxes and keypoints are printed for each frame, annotated
results are saved under ``runs/predict``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

try:  # TensorRT is optional at runtime
    import tensorrt as trt
except Exception:  # pragma: no cover - TensorRT may be missing
    trt = None

import cv2  # type: ignore
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to .engine file")
    ap.add_argument("--source", required=True, help="image/video path or directory")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--device", default=0, help="CUDA device")
    ap.add_argument("--save", action="store_true", help="save annotated outputs")
    ap.add_argument("--show", action="store_true", help="display predictions")
    ap.add_argument("--task", default="pose", help="ultralytics task type")
    return ap.parse_args()


def engine_channels(engine_path: str) -> int:
    """Return input channel dimension for the TensorRT engine."""
    if trt is None:
        return 3
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    shape = engine.get_binding_shape(0)
    return shape[1] if len(shape) >= 4 else 3


def mono_sources(src: str) -> Iterable:
    """Yield grayscale frames from an image, directory or video."""
    p = Path(src)
    if p.is_dir():
        for im_path in sorted(p.iterdir()):
            img = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                yield img
    elif p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}:
        cap = cv2.VideoCapture(str(p))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cap.release()
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(src)
        yield img


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.engine):
        raise SystemExit(f"Engine not found: {args.engine}")

    c_dim = engine_channels(args.engine)
    model = YOLO(args.engine, task=args.task)

    if c_dim == 1:
        print("[INFO] engine expects single-channel input; converting source to grayscale")
        src_iter: Iterable = mono_sources(args.source)
    else:
        src_iter = args.source

    results = model.predict(
        source=src_iter,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        show=args.show,
        stream=True,
        verbose=False,
    )

    for r in results:
        boxes = r.boxes
        kpts = getattr(r, "keypoints", None)
        if boxes is not None and len(boxes):
            print("Boxes (xyxy):", boxes.xyxy.cpu().numpy().tolist())
        if kpts is not None and len(kpts):
            print("Keypoints (xy):", kpts.xy.cpu().numpy().tolist())


if __name__ == "__main__":
    main()

