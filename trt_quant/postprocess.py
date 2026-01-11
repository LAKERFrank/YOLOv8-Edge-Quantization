"""Postprocess TensorRT outputs into Ultralytics-style results."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:  # optional for environments without ultralytics
    import torch
    from ultralytics.engine.results import Results
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Results = None  # type: ignore


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def _pick_prediction(outputs: Dict[str, np.ndarray]) -> np.ndarray:
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    candidates = sorted(outputs.items(), key=lambda item: item[1].size, reverse=True)
    return candidates[0][1]


def _scale_boxes(boxes: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    ratio = meta.get("ratio", 1.0)
    pad = meta.get("pad", (0.0, 0.0))
    offset_x = meta.get("offset_x", 0.0)
    offset_y = meta.get("offset_y", 0.0)

    if isinstance(ratio, (tuple, list)):
        gain_w, gain_h = ratio
    else:
        gain_w = gain_h = ratio

    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, [0, 2]] /= gain_w
    boxes[:, [1, 3]] /= gain_h
    boxes[:, [0, 2]] += offset_x
    boxes[:, [1, 3]] += offset_y
    return boxes


def _scale_keypoints(kpts: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    ratio = meta.get("ratio", 1.0)
    pad = meta.get("pad", (0.0, 0.0))
    offset_x = meta.get("offset_x", 0.0)
    offset_y = meta.get("offset_y", 0.0)

    if isinstance(ratio, (tuple, list)):
        gain_w, gain_h = ratio
    else:
        gain_w = gain_h = ratio

    kpts = kpts.copy()
    kpts[..., 0] = (kpts[..., 0] - pad[0]) / gain_w + offset_x
    kpts[..., 1] = (kpts[..., 1] - pad[1]) / gain_h + offset_y
    return kpts


def postprocess_batch(
    outputs: Dict[str, np.ndarray],
    metas: List[dict[str, Any]],
    conf: float,
    iou: float,
    nc: int = 1,
    nkpt: int = 17,
    names: Dict[int, str] | None = None,
    verbose: bool = False,
) -> List[Any]:
    if torch is None or Results is None:
        raise ImportError("ultralytics and torch are required for postprocess")

    pred = _pick_prediction(outputs)
    if pred.ndim == 2:
        pred = pred[None, ...]

    if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
        if verbose:
            print(f"[postprocess] transpose pred from {pred.shape} to (B,N,no)")
        pred = pred.transpose(0, 2, 1)
    elif verbose:
        print(f"[postprocess] pred shape {pred.shape} (no transpose)")

    results: List[Any] = []
    for b in range(pred.shape[0]):
        pred_b = pred[b]
        no = pred_b.shape[1]
        base_cols = 5 + nc + nkpt * 3
        if no != base_cols:
            inferred_nc = no - 5 - nkpt * 3
            if inferred_nc == 0:
                nc_use = 0
            elif inferred_nc > 0:
                nc_use = inferred_nc
            else:
                raise ValueError(f"unexpected prediction columns: {no}")
        else:
            nc_use = nc

        if verbose:
            print(f"[postprocess] batch {b} columns={no} nc={nc_use}")

        boxes = pred_b[:, :4]
        obj = pred_b[:, 4]
        cls = pred_b[:, 5:5 + nc_use] if nc_use else np.ones((pred_b.shape[0], 1), dtype=pred_b.dtype)
        kpts = pred_b[:, 5 + nc_use:]

        scores = obj * cls.max(1)
        keep = scores >= conf
        boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]

        boxes = xywh2xyxy(boxes)
        kpts = kpts.reshape(-1, nkpt, 3)

        keep_idx = nms(boxes, scores, iou)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        kpts = kpts[keep_idx]

        meta = metas[b]
        boxes = _scale_boxes(boxes, meta)
        kpts = _scale_keypoints(kpts, meta)

        if boxes.size:
            cls_ids = np.zeros((boxes.shape[0], 1), dtype=np.float32)
            boxes_out = np.concatenate([boxes, scores[:, None], cls_ids], axis=1)
            kpts_out = kpts.astype(np.float32)
        else:
            boxes_out = np.zeros((0, 6), dtype=np.float32)
            kpts_out = np.zeros((0, nkpt, 3), dtype=np.float32)

        orig_img = meta.get("orig_img")
        if orig_img is None:
            h, w = meta.get("orig_shape", (0, 0))
            orig_img = np.zeros((h, w, 3), dtype=np.uint8)

        result = Results(
            orig_img=orig_img,
            path=None,
            names=names,
            boxes=torch.from_numpy(boxes_out),
            keypoints=torch.from_numpy(kpts_out),
        )
        results.append(result)

    return results
