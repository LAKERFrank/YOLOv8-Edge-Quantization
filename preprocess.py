from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


def _letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    height, width = image.shape[:2]
    new_h, new_w = new_shape
    ratio = min(new_h / height, new_w / width)
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, (pad_left, pad_top)


def preprocess_3(
    frames3: Sequence[np.ndarray],
    imgsz: Tuple[int, int],
    to_gray: bool = False,
    letterbox: bool = True,
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if len(frames3) != 3:
        raise ValueError("frames3 must contain exactly 3 frames")
    metas: List[Dict[str, Any]] = []
    processed: List[np.ndarray] = []
    for frame in frames3:
        orig_shape = frame.shape[:2]
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[:, :, None]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if letterbox:
            resized, ratio, pad = _letterbox(frame, imgsz)
            input_shape = resized.shape[:2]
        else:
            resized = cv2.resize(frame, imgsz[::-1], interpolation=cv2.INTER_LINEAR)
            ratio = min(imgsz[0] / orig_shape[0], imgsz[1] / orig_shape[1])
            pad = (0, 0)
            input_shape = resized.shape[:2]
        resized = resized.astype(np.float32) / 255.0
        if resized.ndim == 2:
            resized = resized[:, :, None]
        if resized.shape[2] == 3:
            resized = (resized - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        else:
            resized = (resized - mean[0]) / std[0]
        resized = resized.transpose(2, 0, 1)
        processed.append(resized)
        metas.append(
            {
                "orig_shape": orig_shape,
                "input_shape": input_shape,
                "ratio": ratio,
                "pad": pad,
                "offset_x": 0,
                "offset_y": 0,
            }
        )
    x = np.stack(processed, axis=0).astype(np.float32)
    x = np.ascontiguousarray(x)
    return x, metas
