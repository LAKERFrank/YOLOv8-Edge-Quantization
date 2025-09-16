#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build INT8 TensorRT engine from a 1-ch ONNX, using MinMax or Entropy calibrator.
- Supports FP16 fallback for selected layers by name keywords.
- Calibration images must reflect deployment conditions (dark court, far players).
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional

import cv2 as cv
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ---------- Simple, consistent preprocessing ----------

def ensure_gray_u8(img: np.ndarray | None) -> np.ndarray:
    """Convert the given image to uint8 grayscale."""
    if img is None:
        raise ValueError("read image failed")
    x = img
    if x.ndim == 3 and x.shape[2] == 3:
        x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
    if x.dtype in (np.float32, np.float64):
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    return x


def letterbox(gray_u8: np.ndarray, new_shape: tuple[int, int] = (640, 640), color: int = 0,
              scaleup: bool = True) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """Minimal letterbox, grayscale version; returns padded image and (ratio, pad)."""
    shape = gray_u8.shape[:2]  # (h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    img = cv.resize(gray_u8, new_unpad, interpolation=cv.INTER_LINEAR)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return img, (r, r), (left, top)


def preprocess_to_nchw(gray_u8: np.ndarray, imgsz: int = 640, norm: bool = True) -> np.ndarray:
    """Ensure 1ch, letterbox to imgsz, to float32 NCHW [1,1,H,W] in 0~1."""
    g = ensure_gray_u8(gray_u8)
    g, _, _ = letterbox(g, (imgsz, imgsz))
    x = g.astype(np.float32)
    if norm:
        x /= 255.0
    return x[None, None, ...]  # [1,1,H,W]


# ---------- Calibration stream & calibrators ----------


class ImageStream:
    """Iterates calibration images and yields NCHW float32 batches (batch=1)."""

    def __init__(self, root: str, imgsz: int = 640):
        self.paths: List[str] = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")
        self.imgsz = imgsz
        self.reset()

    def reset(self) -> None:
        self.i = 0

    def get_batch(self) -> Optional[np.ndarray]:
        if self.i >= len(self.paths):
            return None
        p = self.paths[self.i]
        self.i += 1
        img = cv.imread(p, cv.IMREAD_UNCHANGED)  # allow color/gray
        return preprocess_to_nchw(img, imgsz=self.imgsz)


class BaseCalibrator:
    def __init__(self, stream: ImageStream, cache_path: str):
        self.stream = stream
        self.cache_path = cache_path
        # Allocate one device buffer for a single batch
        sample = self.stream.get_batch()
        if sample is None:
            raise RuntimeError("Calibration stream is empty")
        self.stream.reset()
        self.device_input = cuda.mem_alloc(sample.nbytes)
        self.host_shape = sample.shape
        self.host_dtype = sample.dtype

    def get_batch_size(self) -> int:
        return self.host_shape[0]  # 1

    def get_batch(self, names: Iterable[str]) -> Optional[List[int]]:  # type: ignore[override]
        batch = self.stream.get_batch()
        if batch is None:
            return None
        assert batch.shape == self.host_shape and batch.dtype == self.host_dtype
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        with open(self.cache_path, "wb") as f:
            f.write(cache)


class MinMaxCalibrator(trt.IInt8MinMaxCalibrator, BaseCalibrator):
    def __init__(self, stream: ImageStream, cache_path: str):
        trt.IInt8MinMaxCalibrator.__init__(self)
        BaseCalibrator.__init__(self, stream, cache_path)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2, BaseCalibrator):
    def __init__(self, stream: ImageStream, cache_path: str):
        trt.IInt8EntropyCalibrator2.__init__(self)
        BaseCalibrator.__init__(self, stream, cache_path)


# ---------- Build engine ----------


def set_workspace_size(config: trt.IBuilderConfig, workspace_mib: int) -> None:
    """Set builder workspace in MiB, handling both legacy and TensorRT 10+ APIs."""
    size_bytes = int(workspace_mib) * (1 << 20)

    # TensorRT < 10 uses the max_workspace_size attribute.
    if hasattr(config, "max_workspace_size"):
        try:
            config.max_workspace_size = size_bytes  # type: ignore[attr-defined]
            return
        except AttributeError:
            # Attribute was removed despite being present via getattr, fall back below.
            pass

    # TensorRT 10+ exposes set_memory_pool_limit with MemoryPoolType.WORKSPACE.
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size_bytes)
        return

    raise AttributeError(
        "Unable to set builder workspace size: unsupported TensorRT API surface."
    )


def set_fp16_fallback(network: trt.INetworkDefinition, keywords: str) -> List[str]:
    """Set precision of layers containing any keyword to FP16 (else INT8 by config)."""
    pinned: List[str] = []
    if not keywords:
        return pinned
    kw = [k.strip() for k in keywords.split(",") if k.strip()]
    if not kw:
        return pinned
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name or ""
        if any(k in name for k in kw):
            layer.precision = trt.DataType.HALF
            for o in range(layer.num_outputs):
                layer.set_output_type(o, trt.DataType.HALF)
            pinned.append(name)
    return pinned


def _as_bytes(serialized_engine) -> bytes:
    """Best-effort conversion of TensorRT serialized outputs to raw bytes."""
    if isinstance(serialized_engine, (bytes, bytearray)):
        return bytes(serialized_engine)
    if hasattr(serialized_engine, "tobytes"):
        return serialized_engine.tobytes()
    try:
        return bytes(memoryview(serialized_engine))
    except TypeError:
        pass
    if hasattr(serialized_engine, "buffer"):
        return bytes(serialized_engine.buffer)
    raise TypeError("Unable to convert serialized engine to bytes")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="path to 1ch ONNX")
    ap.add_argument("--out", required=True, help="output .engine path")
    ap.add_argument("--calib-dir", dest="calib_dir", required=True, help="folder of calibration images")
    ap.add_argument("--imgsz", type=int, default=640, help="network input size (square)")
    ap.add_argument("--calibrator", choices=["minmax", "entropy"], default="minmax")
    ap.add_argument(
        "--fp16-fallback",
        type=str,
        default="model.0,detect",
        help="comma-separated keywords of layers to pin to FP16",
    )
    ap.add_argument("--workspace", type=int, default=4096, help="builder workspace MiB")
    ap.add_argument(
        "--enable-fp16",
        dest="enable_fp16",
        action="store_true",
        help="also enable FP16 (for fallback and fast kernels)",
    )
    ap.add_argument(
        "--disable-fp16",
        dest="enable_fp16",
        action="store_false",
        help="disable FP16 kernels even if fallback layers are requested",
    )
    ap.set_defaults(enable_fp16=True)
    return ap.parse_args()


def build_engine(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")
    if not os.path.isdir(args.calib_dir):
        raise FileNotFoundError(f"Calibration directory not found: {args.calib_dir}")

    stream = ImageStream(args.calib_dir, imgsz=args.imgsz)
    print(f"Calibration images: {len(stream.paths)}")
    cache = os.path.splitext(args.out)[0] + ".calib.cache"
    if args.calibrator == "minmax":
        calibrator = MinMaxCalibrator(stream, cache)
    else:
        calibrator = EntropyCalibrator(stream, cache)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder:
        with builder.create_network(network_flags) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                builder.create_builder_config() as config:

            if hasattr(builder, "max_batch_size"):
                # Older TensorRT versions expect this property even in explicit batch mode,
                # whereas newer releases (TensorRT 10+) removed it entirely.
                # Guard the assignment so both APIs are supported.
                try:
                    builder.max_batch_size = 1  # type: ignore[assignment]
                except AttributeError:
                    pass

            set_workspace_size(config, args.workspace)

            with open(args.onnx, "rb") as f:
                onnx_bytes = f.read()
            if not parser.parse(onnx_bytes):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

            # INT8 + optional FP16
            config.set_flag(trt.BuilderFlag.INT8)
            if args.enable_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            config.int8_calibrator = calibrator

            # Pin sensitive layers to FP16 (stem / detect by default)
            pinned_layers = set_fp16_fallback(network, args.fp16_fallback)
            if pinned_layers:
                print("Pinned layers to FP16:")
                for name in pinned_layers:
                    print(f"  - {name}")

            # (Optional) print layer names to help tuning
            print("== Network layers ==")
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                print(f"{i:03d} | {layer.name} | {layer.type}")

            serialized = None
            if hasattr(builder, "build_serialized_network"):
                # TensorRT 10+ returns an IHostMemory serialized engine directly.
                serialized = builder.build_serialized_network(network, config)
                if serialized is None:
                    raise RuntimeError("build_serialized_network returned None")
            elif hasattr(builder, "build_engine"):
                # Older TensorRT exposes build_engine which returns an ICudaEngine.
                engine = builder.build_engine(network, config)
                if engine is None:
                    raise RuntimeError("build_engine returned None")
                serialized = engine.serialize()
            else:
                raise AttributeError("Unsupported TensorRT builder API: missing build methods")

            engine_bytes = _as_bytes(serialized)

            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "wb") as f:
                f.write(engine_bytes)
            print(f"Saved engine to: {args.out}")


def main() -> None:
    args = parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()
