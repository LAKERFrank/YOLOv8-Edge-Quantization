#!/usr/bin/env python3
import os
import glob
import random
from pathlib import Path

import cv2
import yaml
import numpy as np
import torch
import tensorrt as trt

# =========================
# Config
# =========================
DATA_YAML = os.environ.get(
    "DATA_YAML",
    "/workspaces/CameraSensor/Quantization/pose-dataset/datasets/office-upright/data.yaml",
)

IMGSZ = int(os.environ.get("IMGSZ", "640"))
DEVICE = os.environ.get("DEVICE", "0")
WS_GIB = int(os.environ.get("WS", "8"))

CALIB_BATCH = int(os.environ.get("CALIB_BATCH", "1"))
CALIB_FRAC = float(os.environ.get("CALIB_FRAC", "0.5"))
SEED = int(os.environ.get("SEED", "0"))

OUT_DIR = Path(os.environ.get(
    "OUT_DIR",
    "/workspaces/CameraSensor/LayerSensing/Pose/weights",
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = OUT_DIR / f"pose.onnx"
INT8_ENGINE = OUT_DIR / "int8.engine"
FP16_ENGINE = OUT_DIR / "fp16.engine"
FP32_ENGINE = OUT_DIR / "fp32.engine"

# avoid reusing old TRT8 cache
CALIB_CACHE = OUT_DIR / f"calib.cache"

BUILD_FP32 = os.environ.get("BUILD_FP32", "0") == "1"
BUILD_FP16 = os.environ.get("BUILD_FP16", "0") == "1"
BUILD_INT8 = os.environ.get("BUILD_INT8", "1") == "1"

# Optional: build version-compatible engine
VERSION_COMPATIBLE = os.environ.get("VERSION_COMPATIBLE", "0") == "1"


def log(msg: str):
    print(msg, flush=True)


def set_workspace(config, ws_gib: int):
    ws_bytes = int(ws_gib) * (1 << 30)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)


def letterbox_gray(im, new=640, color=114):
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (new - nh) // 2
    bottom = new - nh - top
    left = (new - nw) // 2
    right = new - nw - left

    return cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )


def preprocess_1ch(path: str, imgsz: int):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = letterbox_gray(im, imgsz)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    x = im.astype(np.float32) / 255.0
    x = x[None, None, :, :]  # (1,1,H,W)
    return x


def deserialize_ok(engine_path: Path, logger_level=trt.Logger.WARNING):
    logger = trt.Logger(logger_level)
    runtime = trt.Runtime(logger)
    if VERSION_COMPATIBLE:
        runtime.engine_host_code_allowed = True
    blob = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(blob)
    return engine is not None, len(blob)


def hostmem_to_bytes(host_mem):
    return bytes(host_mem)


def collect_calib_images(data_yaml: str, frac: float, seed: int, batch_size: int):
    data = yaml.safe_load(open(data_yaml, "r"))
    train_root = data.get("train", None)
    if train_root is None:
        raise RuntimeError("data.yaml missing 'train' field")

    if not str(train_root).startswith("/"):
        train_root = str(Path(data_yaml).parent / train_root)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for ext in exts:
        files += glob.glob(str(Path(train_root) / "**" / f"*{ext}"), recursive=True)
    files = sorted(files)

    if not files:
        raise RuntimeError(f"No images found under train path: {train_root}")

    random.seed(seed)
    k = max(batch_size, int(len(files) * frac))
    k = min(k, len(files))
    return random.sample(files, k)


class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, img_paths, batch_size, cache_file: Path, device: torch.device, imgsz: int):
        super().__init__()
        self.img_paths = img_paths
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.device = device
        self.imgsz = imgsz
        self.index = 0
        self.dev_tensor = None  # keep alive between calls

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index + self.batch_size > len(self.img_paths):
            return None

        batch_paths = self.img_paths[self.index:self.index + self.batch_size]
        host = [preprocess_1ch(p, self.imgsz) for p in batch_paths]
        host = np.concatenate(host, axis=0)  # (B,1,H,W)

        self.dev_tensor = torch.from_numpy(host).to(self.device, non_blocking=True)
        self.index += self.batch_size
        return [int(self.dev_tensor.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            log(f"[INT8] reuse cache: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(cache)
        log(f"[INT8] wrote cache: {self.cache_file}")


def build_engine(out_path: Path, fp16=False, int8=False, calib_files=None):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"Missing ONNX: {ONNX_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(int(DEVICE))

    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 10: no implicit batch path
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = ONNX_PATH.read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    set_workspace(config, WS_GIB)

    if VERSION_COMPATIBLE:
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        log("[TRT] VERSION_COMPATIBLE enabled")

    inp = network.get_input(0)
    log(f"[TRT] input name={inp.name} shape={tuple(inp.shape)} dtype={inp.dtype}")

    # add profile only when ONNX input is dynamic
    if any(d == -1 for d in tuple(inp.shape)):
        profile = builder.create_optimization_profile()
        max_b = max(1, CALIB_BATCH)
        profile.set_shape(
            inp.name,
            (1, 1, IMGSZ, IMGSZ),
            (1, 1, IMGSZ, IMGSZ),
            (max_b, 1, IMGSZ, IMGSZ),
        )
        config.add_optimization_profile(profile)
        log("[TRT] dynamic profile added")

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log("[TRT] FP16 enabled")
        else:
            log("[WARN] platform_has_fast_fp16=False; continue without FP16")

    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("platform_has_fast_int8=False")

        if not calib_files:
            raise RuntimeError("INT8 requested but calib_files is empty")

        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = TorchEntropyCalibrator(
            img_paths=calib_files,
            batch_size=CALIB_BATCH,
            cache_file=CALIB_CACHE,
            device=torch.device(f"cuda:{DEVICE}"),
            imgsz=IMGSZ,
        )
        log("[TRT] INT8 enabled")

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network returned None")

    out_path.write_bytes(hostmem_to_bytes(plan))

    ok, nbytes = deserialize_ok(out_path)
    log(f"[OK] engine={out_path.name} bytes={nbytes} deserialize_ok={ok}")
    if not ok:
        raise RuntimeError(f"deserialize failed: {out_path}")


def main():
    log(f"[ENV] TensorRT version: {trt.__version__}")
    log(f"[ENV] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"[ENV] CUDA device count: {torch.cuda.device_count()}")
        log(f"[ENV] current device: {torch.cuda.current_device()}")

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"Missing ONNX file: {ONNX_PATH}")

    calib_files = collect_calib_images(
        data_yaml=DATA_YAML,
        frac=CALIB_FRAC,
        seed=SEED,
        batch_size=CALIB_BATCH,
    )
    log(f"[DATA] calib_images={len(calib_files)}")

    if BUILD_FP32:
        build_engine(FP32_ENGINE, fp16=False, int8=False, calib_files=None)

    if BUILD_FP16:
        build_engine(FP16_ENGINE, fp16=True, int8=False, calib_files=None)

    if BUILD_INT8:
        build_engine(INT8_ENGINE, fp16=False, int8=True, calib_files=calib_files)

    log("[DONE]")


if __name__ == "__main__":
    main()