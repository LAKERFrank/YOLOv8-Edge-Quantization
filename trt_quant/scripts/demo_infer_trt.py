#!/usr/bin/env python3
"""Demo script for TensorRT 3-in/3-out pose inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from trt_quant.api_infer import PoseTRTInfer  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TensorRT 3-in/3-out demo")
    ap.add_argument("--engine", required=True, help="path to TensorRT .engine")
    ap.add_argument("--images", nargs="*", help="three image paths")
    ap.add_argument("--source", help="image directory or video path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--to-gray", action="store_true", help="convert inputs to grayscale")
    ap.add_argument("--nkpt", type=int, default=17)
    return ap.parse_args()


def load_three_frames(args: argparse.Namespace) -> List[np.ndarray]:
    import cv2

    frames: List[np.ndarray] = []
    if args.images:
        if len(args.images) != 3:
            raise ValueError("--images must provide exactly 3 paths")
        for path in args.images:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(path)
            frames.append(img)
        return frames

    if not args.source:
        raise ValueError("Provide --images or --source")

    src = Path(args.source)
    if src.is_dir():
        for img_path in sorted(src.iterdir()):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img)
            if len(frames) == 3:
                break
    else:
        cap = cv2.VideoCapture(str(src))
        while len(frames) < 3:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    if len(frames) != 3:
        raise RuntimeError("Failed to load three frames from source")
    return frames


def main() -> None:
    args = parse_args()
    frames3 = load_three_frames(args)

    infer = PoseTRTInfer(
        args.engine,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        to_gray=args.to_gray,
        nkpt=args.nkpt,
    )
    results = infer.infer_3in3out(frames3, verbose=True)

    print(f"len(results) = {len(results)}")
    for idx, result in enumerate(results):
        orig_shape = getattr(result, "orig_shape", None) or result.orig_img.shape[:2]
        boxes = result.boxes
        kpts = result.keypoints
        num_boxes = len(boxes) if boxes is not None else 0
        num_kpts = len(kpts) if kpts is not None else 0
        print(f"[{idx}] orig_shape={tuple(orig_shape)}, boxes={num_boxes}, kpt_instances={num_kpts}")
        if kpts is not None:
            kpt_xy = kpts.xy
            kpt_conf = getattr(kpts, "conf", None)
            print(f"keypoints.xy shape: {tuple(kpt_xy.shape)}")
            if kpt_conf is not None:
                print(f"keypoints.conf shape: {tuple(kpt_conf.shape)}")


if __name__ == "__main__":
    main()
