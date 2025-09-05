import os
import random
import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

from utils_io import ensure_exists


def load_paths(list_txt, limit=None, shuffle=False):
    paths = []
    with open(list_txt, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            ensure_exists(ln)
            paths.append(ln)
    assert paths, f"No valid paths found in {list_txt}"
    if shuffle:
        random.shuffle(paths)
    if limit:
        paths = paths[:limit]
    return paths


def letterbox(img, new_shape=640, color=114):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full(new_shape, color, dtype=resized.dtype)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas


def preprocess_gray(path, size=640):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, path
    img = letterbox(img, size)
    img = img.astype(np.float32) / 255.0
    img = img[None, None, ...]
    return img


class CalibDataReader1Ch(CalibrationDataReader):
    def __init__(self, list_txt, input_name, size=640, limit=None):
        self.paths = load_paths(list_txt, limit=limit, shuffle=True)
        self.i = 0
        self.input_name = input_name
        self.size = size

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        x = preprocess_gray(self.paths[self.i], self.size)
        self.i += 1
        return {self.input_name: x}
