#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference with a TensorRT engine using Ultralytics runtime.
Works for YOLO-Pose engine produced by export_trt.py.
"""
import argparse, os

try:
    import tensorrt as trt
except Exception:  # pragma: no cover - TRT optional at runtime
    trt = None

from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to .engine")
    ap.add_argument("--source", required=True, help="image/video path or dir")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--save", action="store_true", help="save rendered outputs")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--task", default="pose", help="ultralytics task type")
    args = ap.parse_args()

    def _get_input_channels(engine_path: str) -> int:
        if trt is None:
            return 3
        logger = trt.Logger(trt.Logger.ERROR)
        with trt.Runtime(logger) as rt:
            with open(engine_path, "rb") as f:
                engine = rt.deserialize_cuda_engine(f.read())
            shape = engine.get_binding_shape(0)
            return shape[1] if len(shape) >= 4 else 3

    c_dim = _get_input_channels(args.engine)
    model = YOLO(args.engine, task=args.task)  # dispatch to TensorRT runtime

    if c_dim == 1:
        import cv2

        def load_mono(path: str):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(path)
            return img

        if os.path.isdir(args.source):
            paths = [os.path.join(args.source, p) for p in sorted(os.listdir(args.source))]
            sources = [load_mono(p) for p in paths]
        else:
            sources = [load_mono(args.source)]
        print("[INFO] engine expects single-channel input; converted source to grayscale")
        results = model.predict(source=sources, imgsz=args.imgsz, device=args.device,
                                conf=args.conf, save=args.save, stream=True, verbose=False)
    else:
        results = model.predict(source=args.source, imgsz=args.imgsz, device=args.device,
                                conf=args.conf, save=args.save, stream=True, verbose=False)

    n_imgs = 0
    for r in results:
        n_imgs += 1
        n_boxes = 0 if r.boxes is None else len(r.boxes)
        n_kpts  = 0 if getattr(r, "keypoints", None) is None else len(r.keypoints)
        print(f"[{n_imgs}] boxes={n_boxes}, keypoints={n_kpts}")
    print(f"[OK] Done. processed frames/images: {n_imgs}")

if __name__ == "__main__":
    main()
