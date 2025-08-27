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


def preprocess_pose(image: np.ndarray, shape: Tuple[int, int], channels: int) -> np.ndarray:
    """Resize and normalize image for pose models supporting 1 or 3 channels."""
    img = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)
    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # 1 x H x W
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # 3 x H x W
    return np.expand_dims(img, 0)  # 1 x C x H x W


def letterbox(image: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image to fit in shape (width, height) while preserving aspect ratio."""
    w, h = shape
    h0, w0 = image.shape[:2]
    r = min(w / w0, h / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = w - new_w, h - new_h
    dw /= 2
    dh /= 2
    img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img, r, (left, top)


def preprocess_tracknet(image: np.ndarray, shape: Tuple[int, int], channels: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox, grayscale, and repeat to match TrackNet channel requirements."""
    img, ratio, pad = letterbox(image, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)  # 1 x H x W
    img = np.repeat(img, channels, axis=0)  # C x H x W
    return np.expand_dims(img, 0), ratio, pad  # 1 x C x H x W, ratio, (pad_w, pad_h)


def run_tracknet(sess: ort.InferenceSession, img: np.ndarray) -> Tuple[int, int]:
    """Run tracknet to get shuttlecock coordinates."""
    input_name = sess.get_inputs()[0].name
    _, c, h, w = sess.get_inputs()[0].shape
    x, ratio, pad = preprocess_tracknet(img, (w, h), c)
    pred = sess.run(None, {input_name: x})[0].reshape(-1)
    if pred.max() <= 1.5:  # assume normalized
        x_pad, y_pad = pred[0] * w, pred[1] * h
    else:
        x_pad, y_pad = pred[0], pred[1]
    x_px = int((x_pad - pad[0]) / ratio)
    y_px = int((y_pad - pad[1]) / ratio)
    return x_px, y_px


def run_pose(sess: ort.InferenceSession, img: np.ndarray, conf: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Run pose model and return list of (box, keypoints)."""
    input_name = sess.get_inputs()[0].name
    _, c, h, w = sess.get_inputs()[0].shape
    x = preprocess_pose(img, (w, h), c)
    pred = sess.run(None, {input_name: x})[0]
    pred = np.squeeze(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, pred.shape[0])
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    num_kpt = (pred.shape[1] - 5) // 3
    boxes, scores, kpts = pred[:, :4], pred[:, 4], pred[:, 5:5 + num_kpt * 3]
    people = []
    for box, score, kpt in zip(boxes, scores, kpts):
        if score < conf:
            continue
        kpt = kpt.reshape(num_kpt, 3)
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
        box_i = box.astype(int).tolist()
        kpts_i = kpt[:, :2].astype(int).tolist()
        LOGGER.info("Person %d bbox: %s kpts: %s", idx, box_i, kpts_i)
    draw_pose(img, people)

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), img)
    LOGGER.info(f"Saved result image to {out_path}")


if __name__ == "__main__":
    main()
