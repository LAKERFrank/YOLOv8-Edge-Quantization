#!/usr/bin/env python3
import os
from pathlib import Path

import torch
import onnx
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

# =========================
# Config
# =========================
MODEL_PT = Path(os.environ.get(
    "MODEL_PT",
    "/workspaces/CameraSensor/Quantization/pose-dataset/weights/119-office/weights/best.pt",
))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
DEVICE = os.environ.get("DEVICE", "0")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))

OUT_DIR = Path(os.environ.get(
    "OUT_DIR",
    "/workspaces/CameraSensor/LayerSensing/Pose/weights",
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = OUT_DIR / f"pose_batch{BATCH_SIZE}.onnx"


def log(msg: str):
    print(msg, flush=True)


def patch_head_for_export(model, dev):
    """
    Patch Ultralytics Pose/Detect head for manual torch.onnx.export().
    """
    head = model.model[-1]

    head.export = True
    head.dynamic = False
    head.format = "onnx"

    if hasattr(head, "shape"):
        head.shape = None

    # Pose.forward() calls self.detect(self, x) in this version
    head.detect = Detect.forward

    if not hasattr(head, "anchors") or head.anchors is None:
        head.anchors = torch.zeros(2, 0, device=dev)

    if not hasattr(head, "strides") or head.strides is None or len(head.strides) == 0:
        head.strides = torch.zeros(1, 0, device=dev)

    return head


def main():
    log(f"[INFO] python={os.sys.executable}")
    log(f"[INFO] model={MODEL_PT}")
    log(f"[INFO] onnx={ONNX_PATH}")
    log(f"[INFO] batch_size={BATCH_SIZE}")

    if not MODEL_PT.exists():
        raise FileNotFoundError(f"Missing MODEL_PT: {MODEL_PT}")

    if BATCH_SIZE < 1:
        raise ValueError(f"BATCH_SIZE must be >= 1, got {BATCH_SIZE}")

    dev = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
    log(f"[INFO] device={dev}")

    y = YOLO(str(MODEL_PT), task="pose")
    model = y.model.fuse().eval().to(dev)

    # Force 1-channel metadata
    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml["channels"] = 1
        model.yaml["ch"] = 1

    head = patch_head_for_export(model, dev)

    # Sanity checks
    first = model.model[0]
    if hasattr(first, "conv"):
        log(f"[CHECK] first conv weight shape = {tuple(first.conv.weight.shape)}")
        log(f"[CHECK] first conv in_channels = {first.conv.in_channels}")
        if first.conv.in_channels != 1:
            raise RuntimeError(
                f"This checkpoint is not a 1-channel model. first conv in_channels={first.conv.in_channels}"
            )

    log(f"[CHECK] head_class = {head.__class__.__name__}")
    log(f"[CHECK] head has detect = {hasattr(head, 'detect')}")

    dummy = torch.zeros(BATCH_SIZE, 1, IMGSZ, IMGSZ, device=dev)
    log(f"[CHECK] dummy shape = {tuple(dummy.shape)}")

    # Dry run first
    with torch.no_grad():
        out = model(dummy)

    if isinstance(out, (tuple, list)):
        output_names = [f"output{i}" for i in range(len(out))]
    else:
        output_names = ["output0"]

    log(f"[CHECK] output_names = {output_names}")

    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,   # static batch=BATCH_SIZE, static H/W
        verbose=False,
    )

    if not ONNX_PATH.exists():
        raise RuntimeError(f"ONNX export failed: {ONNX_PATH}")

    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)

    log("[OK] ONNX export success")
    log(f"[OUT] {ONNX_PATH}")


if __name__ == "__main__":
    main()
