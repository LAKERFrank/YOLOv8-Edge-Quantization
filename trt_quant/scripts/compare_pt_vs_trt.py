#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare FP32 ``.pt`` vs INT8 ``.engine`` results (boxes & keypoints).

For each image the script runs the PyTorch model using the Ultralytics
pipeline, but executes the TensorRT engine manually in order to support
single-channel models (e.g. INT8 pose engines built from grayscale
inputs). Detections from the two models are matched greedily by IoU and
MAE/MAX metrics are reported for matched boxes and keypoints.
"""

import argparse
import glob
import os
import numpy as np
from ultralytics import YOLO


def letterbox(im, new_shape):
    """Resize and pad image while meeting stride-multiple constraints."""
    import cv2

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
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (dw, dh)


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
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


def load_engine(engine_path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    return engine, engine.create_execution_context(), trt


def engine_channels(engine, trt_module):
    if hasattr(engine, "get_binding_shape"):
        shape = engine.get_binding_shape(0)
    else:  # TRT >=10
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt_module.TensorIOMode.INPUT:
                shape = engine.get_tensor_shape(name)
                break
    return int(shape[1]) if len(shape) >= 2 else 3


def infer(engine, context, trt_module, img, c_dim, imgsz, conf, iou, nkpt, nc):
    import cv2
    import pycuda.driver as cuda

    if c_dim == 1 and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if c_dim == 3 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
    if out.size % cols != 0:  # engine may omit class probs
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
    return boxes[keep], kpts[keep]

def iou_xyxy(a, b):
    # a,b: [4] (x1,y1,x2,y2)
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def to_xyxy(boxes):
    # Ultralytics r.boxes.xyxy is a tensor
    if boxes is None or len(boxes)==0:
        return np.zeros((0,4), dtype=np.float32)
    return boxes.xyxy.cpu().numpy().astype(np.float32)

def to_kpts(kpts, nkpt=17):
    if kpts is None or len(kpts) == 0:
        return np.zeros((0, nkpt, 2), dtype=np.float32)
    return kpts.xy.cpu().numpy().astype(np.float32)

def match_by_iou(a_xyxy, b_xyxy, thr=0.3):
    # greedy matching
    if len(a_xyxy)==0 or len(b_xyxy)==0:
        return []
    used_b = set()
    pairs = []
    for i, a in enumerate(a_xyxy):
        best_j, best_iou = -1, -1.0
        for j, b in enumerate(b_xyxy):
            if j in used_b: continue
            iou = iou_xyxy(a, b)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= thr:
            pairs.append((i, best_j))
            used_b.add(best_j)
    return pairs

def mae_max(x, y):
    d = np.abs(x - y).reshape(-1)
    if d.size == 0:
        return np.nan, np.nan
    return float(np.mean(d)), float(np.max(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="path to FP32 .pt")
    ap.add_argument("--engine", required=True, help="path to INT8 .engine")
    ap.add_argument("--images", required=True, help="folder of test images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--task", default="detect",
                    help="model task type, e.g. detect, pose, segment, classify, obb")
    ap.add_argument("--nc", type=int, default=1, help="number of classes for TRT model")
    ap.add_argument("--nkpt", type=int, default=17, help="number of keypoints for TRT model")
    args = ap.parse_args()

    imgs = sorted([p for p in glob.glob(os.path.join(args.images, "*")) if os.path.isfile(p)])
    if not imgs:
        raise SystemExit(f"No images in {args.images}")

    m_pt = YOLO(args.pt, task=args.task)

    import cv2
    import pycuda.driver as cuda

    cuda.init()
    dev = cuda.Device(int(args.device))
    ctx = dev.make_context()
    engine, context, trt_module = load_engine(args.engine)
    c_dim = engine_channels(engine, trt_module)
    ctx.pop()  # release so PyTorch can use its own context

    mae_boxes_all, max_boxes_all = [], []
    mae_kpts_all, max_kpts_all = [], []
    skipped = 0

    try:
        for p in imgs:
            r_pt = m_pt.predict(source=p, imgsz=args.imgsz, device=args.device,
                                conf=args.conf, save=False, stream=False, verbose=False)[0]

            frame = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if frame is None:
                skipped += 1
                continue

            ctx.push()
            try:
                b_boxes, b_kpts_full = infer(engine, context, trt_module, frame, c_dim,
                                             args.imgsz, args.conf, args.iou_thr,
                                             args.nkpt, args.nc)
            finally:
                ctx.pop()
            b_kpts = b_kpts_full[..., :2]

            a_boxes = to_xyxy(r_pt.boxes)
            a_kpts = to_kpts(getattr(r_pt, "keypoints", None), args.nkpt)

            pairs = match_by_iou(a_boxes, b_boxes, thr=args.iou_thr)
            if not pairs:
                skipped += 1
                continue

            for ia, ib in pairs:
                mae_b, max_b = mae_max(a_boxes[ia], b_boxes[ib])
                mae_boxes_all.append(mae_b)
                max_boxes_all.append(max_b)
                if len(a_kpts) > ia and len(b_kpts) > ib:
                    mae_k, max_k = mae_max(a_kpts[ia], b_kpts[ib])
                    mae_kpts_all.append(mae_k)
                    max_kpts_all.append(max_k)
    finally:
        ctx.detach()

    def summarize(name, arr_mae, arr_max):
        arr_mae = np.array([x for x in arr_mae if not np.isnan(x)], dtype=np.float32)
        arr_max = np.array([x for x in arr_max if not np.isnan(x)], dtype=np.float32)
        if arr_mae.size == 0:
            print(f"{name}: no matched samples.")
        else:
            print(f"{name}: MAE={arr_mae.mean():.6f}  MAX={arr_max.max():.6f}  (N={arr_mae.size})")

    print("\n[RESULT] FP32 vs INT8 (result-level)")
    summarize("Boxes (xyxy)", mae_boxes_all, max_boxes_all)
    summarize("Keypoints (x,y)", mae_kpts_all, max_kpts_all)
    print(f"Skipped images (no matches): {skipped}")

if __name__ == "__main__":
    main()
