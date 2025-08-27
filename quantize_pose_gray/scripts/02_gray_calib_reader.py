import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader


def load_paths(list_txt):
    with open(list_txt, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def preprocess_gray(path, size=640):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img[None, None, ...]
    return img


class GrayCalibReader(CalibrationDataReader):
    def __init__(self, list_txt, input_name, size=640):
        self.paths = load_paths(list_txt)
        self.i = 0
        self.input_name = input_name
        self.size = size

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        x = preprocess_gray(self.paths[self.i], self.size)
        self.i += 1
        return {self.input_name: x}
