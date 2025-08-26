"""Simple demo for running INT8 ONNX pose and track models.

The script loads a pose model (e.g. YOLOv8 pose) and a tracknet model
for shuttlecock detection. It visualizes the shuttlecock with a red dot
and human keypoints/skeletons, saves the output image, and logs
information such as shuttle coordinates, number of people, bounding
boxes, and keypoints.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# Skeleton definition for COCO 17 keypoints
# (start_index, end_index)
SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # right arm
    (0, 5), (5, 6), (6, 7),          # left arm
    (5, 11), (6, 12),                # body
    (11, 12), (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),              # right leg
]

LOGGER = logging.getLogger("onnx_int8_demo")


def preprocess(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize and normalize image for RGB models."""
    img = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, 0)


def preprocess_tracknet(image: np.ndarray, shape: Tuple[int, int], channels: int) -> np.ndarray:
    """Resize, grayscale, and repeat to match TrackNet channel requirements."""
    img = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)  # 1 x H x W
    img = np.repeat(img, channels, axis=0)  # C x H x W
    return np.expand_dims(img, 0)  # 1 x C x H x W


def run_tracknet(sess: ort.InferenceSession, img: np.ndarray) -> Tuple[int, int]:
    """Run tracknet to get shuttlecock coordinates."""
    input_name = sess.get_inputs()[0].name
    _, c, h, w = sess.get_inputs()[0].shape
    x = preprocess_tracknet(img, (w, h), c)
    pred = sess.run(None, {input_name: x})[0].reshape(-1)
    h_img, w_img = img.shape[:2]
    if pred.max() <= 1.5:  # assume normalized
        x_px, y_px = int(pred[0] * w_img), int(pred[1] * h_img)
    else:
        x_px, y_px = int(pred[0]), int(pred[1])
    return x_px, y_px


def run_pose(sess: ort.InferenceSession, img: np.ndarray, conf: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Run pose model and return list of (box, keypoints)."""
    input_name = sess.get_inputs()[0].name
    _, _, h, w = sess.get_inputs()[0].shape
    x = preprocess(img, (w, h))
    pred = sess.run(None, {input_name: x})[0][0]
    boxes, scores, kpts = pred[:, :4], pred[:, 4], pred[:, 5:]
    people = []
    for box, score, kpt in zip(boxes, scores, kpts):
        if score < conf:
            continue
        kpt = kpt.reshape(-1, 3)
        people.append((box, kpt))
    return people


def draw_pose(image: np.ndarray, people: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Draw bounding boxes and skeletons on the image."""
    for box, kpts in people:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x, y, _ in kpts:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), -1)
        for i, j in SKELETON:
            xi, yi, _ = kpts[i]
            xj, yj, _ = kpts[j]
            cv2.line(image, (int(xi), int(yi)), (int(xj), int(yj)), (255, 0, 0), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="INT8 ONNX pose and tracknet demo")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--pose",
        default=str(root / "onnx/yolov8n-pose-int8.onnx"),
        help="Path to pose ONNX model",
    )
    ap.add_argument(
        "--track",
        default=str(root / "onnx/tracknet1000-int8.onnx"),
        help="Path to tracknet ONNX model",
    )
    ap.add_argument(
        "--image",
        default=str(root / "val/pose/1_00_01_00349.jpg"),
        help="Input image path",
    )
    ap.add_argument("--output", default="test.jpg", help="Output image path")
    ap.add_argument("--conf", type=float, default=0.25, help="Pose confidence threshold")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    pose_sess = ort.InferenceSession(args.pose, providers=["CPUExecutionProvider"])
    track_sess = ort.InferenceSession(args.track, providers=["CPUExecutionProvider"])

    ball_x, ball_y = run_tracknet(track_sess, img)
    LOGGER.info(f"Shuttlecock at: ({ball_x}, {ball_y})")
    cv2.circle(img, (ball_x, ball_y), 5, (0, 0, 255), -1)

    people = run_pose(pose_sess, img, args.conf)
    LOGGER.info(f"Detected {len(people)} person(s)")
    for idx, (box, kpt) in enumerate(people):
        LOGGER.info(f"Person {idx} bbox: {box.tolist()} kpts: {kpt[:, :2].tolist()}")
    draw_pose(img, people)

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), img)
    LOGGER.info(f"Saved result image to {out_path}")


if __name__ == "__main__":
    main()
