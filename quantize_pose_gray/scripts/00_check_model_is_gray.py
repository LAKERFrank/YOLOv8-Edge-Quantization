import torch
from ultralytics import YOLO

# Modify to the actual path of the 1-channel YOLOv8-Pose weights
WEIGHT = "yolov8n-pose-gray.pt"

m = YOLO(WEIGHT).model
first = m.model[0].conv  # First Conv2d layer
print("First conv in_channels:", first.in_channels)
assert first.in_channels == 1, "Model is not 1-channel!"

try:
    print("names:", m.names)
except Exception:
    pass
