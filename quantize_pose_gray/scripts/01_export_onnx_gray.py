from ultralytics import YOLO
from pathlib import Path

PT = "yolov8n-pose-gray.pt"
ONNX_OUT = "yolov8n-pose-gray.fp32.onnx"

y = YOLO(PT)
exported = y.export(
    format="onnx",
    opset=13,
    simplify=True,
    imgsz=640,
    dynamic=False,
    half=False,
    verbose=True,
    optimize=True,
)

# Ultralytics writes under runs/, move or rename as needed to ONNX_OUT
Path(exported).replace(ONNX_OUT)
