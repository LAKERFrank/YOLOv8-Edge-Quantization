# YOLOv8 Quantization (Medium Article)

This directory provides scripts that follow the Medium article ["Quantizing YOLOv8 models"](https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2).

## Requirements

Both scripts expect the following Python packages:

- `ultralytics` for exporting PyTorch weights to ONNX
- `onnxruntime` and `onnxruntime-tools` for quantization
- `opencv-python` for image preprocessing in the static workflow

## Scripts

- `dynamic.py` – Export YOLOv8 weights to ONNX and apply weight-only INT8 dynamic quantization.
- `static.py` – Export YOLOv8 weights to ONNX and apply post-training static quantization using calibration images.

## Usage

### Dynamic quantization
```bash
python dynamic.py --weights path/to/yolov8n.pt --onnx-out yolov8n-fp32.onnx --quant-out yolov8n-int8.onnx
```

### Static quantization
```bash
python static.py --weights path/to/yolov8n.pt --calib-dir path/to/images --onnx-out yolov8n-fp32.onnx --quant-out yolov8n-int8.onnx
```
