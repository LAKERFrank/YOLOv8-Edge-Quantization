import argparse
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from api_infer import PoseTRTInfer


def load_images(paths: Sequence[str]):
    frames = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        frames.append(frame)
    return frames


def load_images_from_dir(img_dir: str) -> List[str]:
    dir_path = Path(img_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    image_paths = sorted(
        str(p)
        for p in dir_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {img_dir}")
    return image_paths


def chunked(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _pairwise_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    inter_x1 = np.maximum(x11[:, None], x21[None])
    inter_y1 = np.maximum(y11[:, None], y21[None])
    inter_x2 = np.minimum(x12[:, None], x22[None])
    inter_y2 = np.minimum(y12[:, None], y22[None])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    area1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    union = area1[:, None] + area2[None] - inter
    return inter / np.maximum(union, 1e-6)


def _match_by_iou(boxes1: np.ndarray, boxes2: np.ndarray, iou_thresh: float = 0.5):
    ious = _pairwise_iou(boxes1, boxes2)
    matches = []
    for i in range(ious.shape[0]):
        j = int(np.argmax(ious[i])) if ious.shape[1] else -1
        if j >= 0 and ious[i, j] >= iou_thresh:
            matches.append((i, j))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine")
    parser.add_argument("--weights", required=True, help="Path to PyTorch weights")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--img", nargs=3, help="Three image paths")
    group.add_argument("--img-dir", help="Directory containing images to process in batches of 3")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    imgsz = tuple(args.imgsz)

    torch_model = YOLO(args.weights)
    trt_infer = PoseTRTInfer(args.engine, conf=args.conf, iou=args.iou, imgsz=imgsz)
    if args.img_dir:
        image_paths = load_images_from_dir(args.img_dir)
        batches = chunked(image_paths, 3)
        if len(batches[-1]) < 3:
            print(
                f"[validate] warning: dropping last {len(batches[-1])} image(s) to keep batch=3"
            )
            batches = batches[:-1]
        for batch_idx, batch_paths in enumerate(batches):
            frames3 = load_images(batch_paths)
            torch_results = torch_model.predict(
                frames3, imgsz=imgsz, conf=args.conf, iou=args.iou
            )
            trt_results = trt_infer.infer_3in3out(frames3)
            for idx, (torch_res, trt_res) in enumerate(zip(torch_results, trt_results)):
                torch_boxes = (
                    torch_res.boxes.xyxy.cpu().numpy() if torch_res.boxes else np.zeros((0, 4))
                )
                trt_boxes = trt_res.boxes
                print(
                    f"[batch {batch_idx} frame {idx}] torch boxes: {torch_boxes.shape[0]}, "
                    f"trt boxes: {trt_boxes.shape[0]}"
                )
                matches = _match_by_iou(torch_boxes, trt_boxes)
                if not matches:
                    print("  no matches for keypoints comparison")
                    continue
                torch_kpts = torch_res.keypoints.xy.cpu().numpy() if torch_res.keypoints else None
                trt_kpts = trt_res.keypoints.xy if trt_res.keypoints else None
                if torch_kpts is None or trt_kpts is None:
                    print("  missing keypoints in one of the results")
                    continue
                errors = []
                for i, j in matches:
                    t_kpt = torch_kpts[i]
                    r_kpt = trt_kpts[j]
                    errors.append(np.linalg.norm(t_kpt - r_kpt, axis=1).mean())
                if errors:
                    print(f"  mean keypoint pixel error: {np.mean(errors):.4f}")
                torch_conf = (
                    torch_res.boxes.conf.cpu().numpy() if torch_res.boxes else np.array([])
                )
                trt_conf = trt_res.scores
                if torch_conf.size and trt_conf.size:
                    print(
                        f"  conf mean (torch/trt): {torch_conf.mean():.4f}/"
                        f"{trt_conf.mean():.4f}"
                    )
    else:
        frames3 = load_images(args.img)
        torch_results = torch_model.predict(frames3, imgsz=imgsz, conf=args.conf, iou=args.iou)
        trt_results = trt_infer.infer_3in3out(frames3)
        for idx, (torch_res, trt_res) in enumerate(zip(torch_results, trt_results)):
            torch_boxes = torch_res.boxes.xyxy.cpu().numpy() if torch_res.boxes else np.zeros((0, 4))
            trt_boxes = trt_res.boxes
            print(f"[frame {idx}] torch boxes: {torch_boxes.shape[0]}, trt boxes: {trt_boxes.shape[0]}")
            matches = _match_by_iou(torch_boxes, trt_boxes)
            if not matches:
                print("  no matches for keypoints comparison")
                continue
            torch_kpts = torch_res.keypoints.xy.cpu().numpy() if torch_res.keypoints else None
            trt_kpts = trt_res.keypoints.xy if trt_res.keypoints else None
            if torch_kpts is None or trt_kpts is None:
                print("  missing keypoints in one of the results")
                continue
            errors = []
            for i, j in matches:
                t_kpt = torch_kpts[i]
                r_kpt = trt_kpts[j]
                errors.append(np.linalg.norm(t_kpt - r_kpt, axis=1).mean())
            if errors:
                print(f"  mean keypoint pixel error: {np.mean(errors):.4f}")
            torch_conf = torch_res.boxes.conf.cpu().numpy() if torch_res.boxes else np.array([])
            trt_conf = trt_res.scores
            if torch_conf.size and trt_conf.size:
                print(
                    f"  conf mean (torch/trt): {torch_conf.mean():.4f}/{trt_conf.mean():.4f}"
                )


if __name__ == "__main__":
    main()
