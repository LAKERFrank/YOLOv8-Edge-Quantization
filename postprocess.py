import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class Keypoints:
    xy: np.ndarray
    conf: Optional[np.ndarray] = None


@dataclasses.dataclass
class Result:
    boxes: np.ndarray
    scores: np.ndarray
    keypoints: Optional[Keypoints]
    orig_shape: Tuple[int, int]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _select_prediction_output(outputs: Dict[str, np.ndarray]) -> np.ndarray:
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    best_name = None
    best_size = -1
    for name, arr in outputs.items():
        size = arr.size
        if size > best_size:
            best_size = size
            best_name = name
    if best_name is None:
        raise ValueError("No outputs available for postprocess")
    return outputs[best_name]


def _scale_boxes(
    boxes: np.ndarray, meta: Dict[str, Any]
) -> np.ndarray:
    pad_w, pad_h = meta.get("pad", (0, 0))
    ratio = meta.get("ratio", 1.0)
    offset_x = meta.get("offset_x", 0)
    offset_y = meta.get("offset_y", 0)
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= ratio
    boxes[:, [0, 2]] += offset_x
    boxes[:, [1, 3]] += offset_y
    return boxes


def _scale_keypoints(
    kpts: np.ndarray, meta: Dict[str, Any]
) -> np.ndarray:
    pad_w, pad_h = meta.get("pad", (0, 0))
    ratio = meta.get("ratio", 1.0)
    offset_x = meta.get("offset_x", 0)
    offset_y = meta.get("offset_y", 0)
    kpts[..., 0] = (kpts[..., 0] - pad_w) / ratio + offset_x
    kpts[..., 1] = (kpts[..., 1] - pad_h) / ratio + offset_y
    return kpts


def postprocess_batch(
    outputs: Dict[str, np.ndarray],
    metas: List[Dict[str, Any]],
    conf: float,
    iou: float,
    num_kpts: int = 17,
    verbose: bool = True,
) -> List[Result]:
    pred = _select_prediction_output(outputs)
    if verbose:
        print(f"[postprocess] raw pred shape: {pred.shape}")
    if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
        if verbose:
            print("[postprocess] transpose pred from (B,no,n) to (B,n,no)")
        pred = pred.transpose(0, 2, 1)
    if pred.ndim != 3:
        raise ValueError("Prediction output must be 3D (B,N,NO)")
    batch_size = pred.shape[0]
    results: List[Result] = []
    for b in range(batch_size):
        pred_b = pred[b]
        if verbose:
            print(f"[postprocess] batch {b} pred shape: {pred_b.shape}")
        if pred_b.shape[1] < 5:
            raise ValueError("Prediction output must have at least 5 columns")
        scores = pred_b[:, 4]
        keep_mask = scores >= conf
        pred_b = pred_b[keep_mask]
        scores = scores[keep_mask]
        boxes = _xywh_to_xyxy(pred_b[:, :4]) if pred_b.size else np.zeros((0, 4))
        kpts = None
        kpt_conf = None
        extra = pred_b[:, 5:]
        if extra.size and (extra.shape[1] % 3 == 0):
            kpt_dim = 3
            kpts = extra.reshape(-1, extra.shape[1] // kpt_dim, kpt_dim)
            kpt_conf = kpts[..., 2]
            kpts = kpts[..., :2]
        elif extra.size and (extra.shape[1] % 2 == 0):
            kpt_dim = 2
            kpts = extra.reshape(-1, extra.shape[1] // kpt_dim, kpt_dim)
        if boxes.size:
            keep_indices = _nms(boxes, scores, iou)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            if kpts is not None:
                kpts = kpts[keep_indices]
                if kpt_conf is not None:
                    kpt_conf = kpt_conf[keep_indices]
        if boxes.size:
            boxes = _scale_boxes(boxes, metas[b])
        if kpts is not None and kpts.size:
            kpts = _scale_keypoints(kpts, metas[b])
        keypoints = None
        if kpts is not None:
            keypoints = Keypoints(xy=kpts, conf=kpt_conf)
        results.append(
            Result(
                boxes=boxes,
                scores=scores,
                keypoints=keypoints,
                orig_shape=metas[b]["orig_shape"],
            )
        )
    return results
