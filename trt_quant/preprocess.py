"""Preprocessing helpers for 3-frame batch inference."""
from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

try:  # optional dependency for --help
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _letterbox(im: np.ndarray, new_shape: Tuple[int, int], color: Tuple[int, int, int]) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    if cv2 is None:
        raise ImportError("opencv-python is required for letterbox")
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess_3(
    frames3: Sequence[np.ndarray],
    imgsz: int,
    to_gray: bool = False,
    letterbox: bool = True,
    bgr_to_rgb: bool = True,
    mean: Iterable[float] | None = None,
    std: Iterable[float] | None = None,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
    roi: Sequence[Tuple[int, int, int, int]] | None = None,
) -> Tuple[np.ndarray, List[dict[str, Any]]]:
    """Preprocess three frames into NCHW batch.

    Returns:
        x: np.float32 array of shape (3, C, H, W)
        metas: list of per-frame metadata for scale-back
    """
    if len(frames3) != 3:
        raise ValueError("frames3 must contain exactly 3 frames")
    if cv2 is None:
        raise ImportError("opencv-python is required for preprocessing")

    mean_arr = None if mean is None else np.array(list(mean), dtype=np.float32)
    std_arr = None if std is None else np.array(list(std), dtype=np.float32)

    imgs: List[np.ndarray] = []
    metas: List[dict[str, Any]] = []

    for idx, frame in enumerate(frames3):
        if frame is None:
            raise ValueError(f"frame {idx} is None")
        orig_shape = frame.shape[:2]
        offset_x = offset_y = 0
        if roi is not None:
            x0, y0, w0, h0 = roi[idx]
            frame = frame[y0 : y0 + h0, x0 : x0 + w0]
            offset_x, offset_y = x0, y0

        if to_gray and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif not to_gray and frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if letterbox:
            resized, ratio, pad = _letterbox(frame, (imgsz, imgsz), pad_color)
        else:
            resized = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            ratio = (imgsz / frame.shape[1], imgsz / frame.shape[0])
            pad = (0.0, 0.0)

        if to_gray:
            if resized.ndim == 2:
                resized = resized[:, :, None]
        else:
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            if bgr_to_rgb:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        resized = resized.astype(np.float32) / 255.0
        if mean_arr is not None:
            resized = resized - mean_arr
        if std_arr is not None:
            resized = resized / std_arr

        resized = np.transpose(resized, (2, 0, 1))
        imgs.append(resized)

        metas.append(
            {
                "orig_shape": orig_shape,
                "input_shape": (imgsz, imgsz),
                "ratio": ratio,
                "pad": pad,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "orig_img": frame,
            }
        )

    x = np.stack(imgs, axis=0).astype(np.float32)
    return np.ascontiguousarray(x), metas
