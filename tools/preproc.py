import cv2
import numpy as np

def letterbox_1ch(img, new_shape=(640, 640), color=114):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full(new_shape, color, dtype=img_resized.dtype)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    canvas[top:top+nh, left:left+nw] = img_resized
    return canvas, r, (left, top)

def preproc_1ch_letterbox(path, size=640):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, path
    lb, _, _ = letterbox_1ch(img, (size, size), color=114)
    x = lb.astype(np.float32) / 255.0
    x = x[None, None, ...]
    return x
