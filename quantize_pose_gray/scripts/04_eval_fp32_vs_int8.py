import cv2
import numpy as np
import onnxruntime as ort
import importlib

_reader = importlib.import_module("02_gray_calib_reader")
load_paths = _reader.load_paths
preprocess_gray = _reader.preprocess_gray

FP32_ONNX = "yolov8n-pose-gray.fp32.onnx"
INT8_ONNX = "yolov8n-pose-gray.int8.qdq.onnx"
VAL_LIST = "data/val_list.txt"
IMG_SIZE = 640
N_SAMPLES = 200


def run_session(sess, x):
    out_names = [o.name for o in sess.get_outputs()]
    return sess.run(out_names, {sess.get_inputs()[0].name: x})[0]


def main():
    val_paths = load_paths(VAL_LIST)[:N_SAMPLES]
    s0 = ort.InferenceSession(FP32_ONNX, providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"])

    diffs = []
    for p in val_paths:
        x = preprocess_gray(p, IMG_SIZE)
        y0 = run_session(s0, x)
        y1 = run_session(s1, x)

        y0 = np.asarray(y0, dtype=np.float32).reshape(-1)
        y1 = np.asarray(y1, dtype=np.float32).reshape(-1)
        ma = np.mean(np.abs(y0 - y1))
        mx = np.max(np.abs(y0 - y1))
        diffs.append([ma, mx])

    diffs = np.asarray(diffs, dtype=np.float32)
    print(f"[A/B] mean(|Δ|)={diffs[:,0].mean():.6f}   max(|Δ|)={diffs[:,1].max():.6f}")


if __name__ == "__main__":
    main()
