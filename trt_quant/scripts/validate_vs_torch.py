#!/usr/bin/env python3
"""Validate TensorRT vs PyTorch results on 3-frame batches."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from trt_quant.api_infer import PoseTRTInfer  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate TensorRT vs PyTorch outputs")
    ap.add_argument("--pt", required=True, help="path to PyTorch .pt")
    ap.add_argument("--engine", required=True, help="path to TensorRT .engine")
    ap.add_argument("--images", nargs="*", help="three image paths")
    ap.add_argument("--source", help="image directory or video path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--to-gray", action="store_true")
    ap.add_argument("--nkpt", type=int, default=17)
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
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


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def match_by_iou(a_xyxy: np.ndarray, b_xyxy: np.ndarray, thr: float = 0.3) -> List[Tuple[int, int]]:
    if len(a_xyxy) == 0 or len(b_xyxy) == 0:
        return []
    used_b = set()
    pairs: List[Tuple[int, int]] = []
    for i, a in enumerate(a_xyxy):
        best_j, best_iou = -1, -1.0
        for j, b in enumerate(b_xyxy):
            if j in used_b:
                continue
            iou = iou_xyxy(a, b)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= thr:
            pairs.append((i, best_j))
            used_b.add(best_j)
    return pairs


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    args = parse_args()
    frames3 = load_three_frames(args)

    model = YOLO(args.pt)
    pt_results = model.predict(frames3, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)

    trt_infer = PoseTRTInfer(
        args.engine,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        to_gray=args.to_gray,
        nkpt=args.nkpt,
        device_index=args.device,
    )
    try:
        trt_results = trt_infer.infer_3in3out(frames3)

        for idx, (pt_res, trt_res) in enumerate(zip(pt_results, trt_results)):
            pt_boxes = pt_res.boxes.xyxy.cpu().numpy() if pt_res.boxes is not None else np.zeros((0, 4))
            trt_boxes = trt_res.boxes.xyxy.cpu().numpy() if trt_res.boxes is not None else np.zeros((0, 4))
            pt_kpts = pt_res.keypoints.xy.cpu().numpy() if pt_res.keypoints is not None else np.zeros((0, args.nkpt, 2))
            trt_kpts = trt_res.keypoints.xy.cpu().numpy() if trt_res.keypoints is not None else np.zeros((0, args.nkpt, 2))
            pt_conf = pt_res.boxes.conf.cpu().numpy() if pt_res.boxes is not None else np.zeros((0,))
            trt_conf = trt_res.boxes.conf.cpu().numpy() if trt_res.boxes is not None else np.zeros((0,))

            pairs = match_by_iou(pt_boxes, trt_boxes)
            if pairs:
                pt_kpts_m = np.stack([pt_kpts[i] for i, _ in pairs], axis=0)
                trt_kpts_m = np.stack([trt_kpts[j] for _, j in pairs], axis=0)
                kpt_mae = mean_abs_error(pt_kpts_m, trt_kpts_m)
            else:
                kpt_mae = float("nan")

            conf_diff = mean_abs_error(pt_conf, trt_conf) if pt_conf.size and trt_conf.size else float("nan")

            print(f"[frame {idx}] pt_boxes={len(pt_boxes)} trt_boxes={len(trt_boxes)}")
            print(f"  matched_pairs={len(pairs)} kpt_mae={kpt_mae:.4f}")
            print(f"  conf_mean_abs_diff={conf_diff:.4f}")
    finally:
        trt_infer.close()


if __name__ == "__main__":
    main()
