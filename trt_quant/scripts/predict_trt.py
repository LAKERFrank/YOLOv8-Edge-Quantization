#!/usr/bin/env python3
"""Run TensorRT pose inference without the Ultralytics pipeline.

This script loads a TensorRT engine, performs preprocessing (including
automatic grayscale conversion for single-channel engines), executes
the model on the GPU, and prints bounding boxes and keypoints for each
input frame. Annotated results can optionally be saved or displayed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Tuple

try:  # numpy may be absent when only showing --help
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TensorRT pose inference")
    ap.add_argument("--engine", required=True, help="path to TensorRT .engine file")
    ap.add_argument("--source", required=True, help="image/video path or directory")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    ap.add_argument("--save", action="store_true", help="save annotated outputs")
    ap.add_argument("--show", action="store_true", help="display predictions")
    ap.add_argument("--nc", type=int, default=1, help="number of classes")
    ap.add_argument("--nkpt", type=int, default=17, help="number of keypoints")
    return ap.parse_args()


def letterbox(im: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints."""
    import cv2

    shape = im.shape[:2]  # current shape (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Pure Python NMS."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
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


def load_engine(engine_path: str):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    return engine, engine.create_execution_context(), trt


def engine_channels(engine, trt_module) -> int:
    if hasattr(engine, "get_binding_shape"):
        shape = engine.get_binding_shape(0)
    else:  # TRT >=10
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt_module.TensorIOMode.INPUT:
                shape = engine.get_tensor_shape(name)
                break
    return int(shape[1]) if len(shape) >= 2 else 3


def frames_from_source(src: str) -> Iterator[Tuple[np.ndarray, str | None]]:
    import cv2

    p = Path(src)
    if p.is_dir():
        for img_path in sorted(p.iterdir()):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                yield img, str(img_path.name)
    elif p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}:
        cap = cv2.VideoCapture(str(p))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame, f"frame{idx:05d}.jpg"
            idx += 1
        cap.release()
    else:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(src)
        yield img, p.name


def infer(engine, context, trt_module, img: np.ndarray, c_dim: int, imgsz: int,
          conf: float, iou: float, nkpt: int, nc: int):
    import cv2
    import pycuda.driver as cuda

    if c_dim == 1 and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if c_dim == 3 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    im0 = img.copy()
    if im0.ndim == 2:
        im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
    img, ratio, (dw, dh) = letterbox(img, (imgsz, imgsz))
    if c_dim == 1:
        img = img[..., None]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]

    if hasattr(context, "set_binding_shape"):
        context.set_binding_shape(0, img.shape)
        output_shape = context.get_binding_shape(1)
        dtype_in = np.float32
        dtype_out = np.float32
        bindings = [None] * 2
        in_idx, out_idx = 0, 1
    else:  # TRT >=10
        in_idx = out_idx = None
        for i in range(context.engine.num_io_tensors):
            name = context.engine.get_tensor_name(i)
            mode = context.engine.get_tensor_mode(name)
            if mode == trt_module.TensorIOMode.INPUT:
                in_name, in_idx = name, i
            elif mode == trt_module.TensorIOMode.OUTPUT:
                out_name, out_idx = name, i
        assert in_idx is not None and out_idx is not None
        context.set_input_shape(in_name, img.shape)
        output_shape = context.get_tensor_shape(out_name)
        dtype_in = trt_module.nptype(context.engine.get_tensor_dtype(in_name))
        dtype_out = trt_module.nptype(context.engine.get_tensor_dtype(out_name))
        bindings = [0] * context.engine.num_io_tensors

    d_input = cuda.mem_alloc(img.nbytes)
    out_bytes = int(np.prod(output_shape)) * np.dtype(dtype_out).itemsize
    d_output = cuda.mem_alloc(out_bytes)
    cuda.memcpy_htod(d_input, img.astype(dtype_in))
    bindings[in_idx] = int(d_input)
    bindings[out_idx] = int(d_output)
    context.execute_v2(bindings)
    out = np.empty(output_shape, dtype=dtype_out)
    cuda.memcpy_dtoh(out, d_output)
    d_input.free()
    d_output.free()

    cols = 5 + nc + nkpt * 3
    if out.size % cols != 0:  # engine may omit class probabilities
        nc = 0
        cols = 5 + nkpt * 3
    if out.ndim == 3:
        if out.shape[1] == cols:
            pred = out[0].T
        elif out.shape[2] == cols:
            pred = out[0]
        else:
            pred = out.reshape(-1, cols)
    else:
        pred = out.reshape(-1, cols)
    boxes = pred[:, :4]
    obj = pred[:, 4]
    cls = pred[:, 5:5 + nc] if nc else np.ones((pred.shape[0], 1), dtype=pred.dtype)
    kpts = pred[:, 5 + nc:]
    scores = obj * cls.max(1)
    keep = scores >= conf
    boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
    boxes = xywh2xyxy(boxes)
    boxes -= np.array([dw, dh, dw, dh])
    boxes /= ratio
    kpts = kpts.reshape(-1, nkpt, 3)
    kpts[..., 0] = (kpts[..., 0] - dw) / ratio
    kpts[..., 1] = (kpts[..., 1] - dh) / ratio
    keep = nms(boxes, scores, iou)
    boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]

    for box, score, kp in zip(boxes, scores, kpts):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 1)
        label = f"{float(score):.2f}"
        cv2.putText(im0, label, (x1, max(y1 - 2, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 0), 1, cv2.LINE_AA)
        for x, y, c in kp:
            if c > 0:
                cv2.circle(im0, (int(x), int(y)), 2, (54, 172, 245), -1)
    return im0, boxes, kpts


def main() -> None:
    args = parse_args()
    if np is None:
        raise SystemExit("numpy is required to run this script")
    import pycuda.driver as cuda
    cuda.init()
    dev = cuda.Device(args.device)
    ctx = dev.make_context()
    try:
        import cv2
        engine, context, trt_module = load_engine(args.engine)
        c_dim = engine_channels(engine, trt_module)

        save_dir = Path("runs/predict")
        if args.save:
            save_dir.mkdir(parents=True, exist_ok=True)

        for frame, name in frames_from_source(args.source):
            im, boxes, kpts = infer(engine, context, trt_module, frame, c_dim,
                                    args.imgsz, args.conf, args.iou, args.nkpt, args.nc)
            print("Boxes (xyxy):", boxes.tolist())
            print("Keypoints (xy):", kpts[..., :2].tolist())
            if args.save and name is not None:
                cv2.imwrite(str(save_dir / name), im)
            if args.show:
                cv2.imshow("result", im)
                cv2.waitKey(1)
    finally:
        ctx.pop()
        ctx.detach()


if __name__ == "__main__":
    main()

