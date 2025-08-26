# YOLOv8 - ONNX Runtime

This project implements YOLOv8 using ONNX Runtime.

## Installation

To run this project, you need to install the required dependencies. The following instructions will guide you through the installation process.

### Installing Required Dependencies

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Installing `onnxruntime-gpu`

If you have an NVIDIA GPU and want to leverage GPU acceleration, you can install the onnxruntime-gpu package using the following command:

```bash
pip install onnxruntime-gpu
```

Note: Make sure you have the appropriate GPU drivers installed on your system.

### Installing `onnxruntime` (CPU version)

If you don't have an NVIDIA GPU or prefer to use the CPU version of onnxruntime, you can install the onnxruntime package using the following command:

```bash
pip install onnxruntime
```

### Usage

After successfully installing the required packages, you can run the YOLOv8 implementation using the following command:

```bash
python main.py --model yolov8n.onnx --img image.jpg --conf-thres 0.5 --iou-thres 0.5 --save result.jpg
```

Make sure to replace yolov8n.onnx with the path to your YOLOv8 ONNX model file, image.jpg with the path to your input image, and adjust the confidence threshold (conf-thres) and IoU threshold (iou-thres) values as needed.

Use `--save <path>` to write the annotated image to disk and `--show` to display it in a window (requires GUI support).

### Pose Visualization

When provided a pose model (e.g. `onnx/yolov8n-pose-fp32.onnx`), the script decodes keypoints and draws the skeleton:

```bash
python main.py --model ../../onnx/yolov8n-pose-fp32.onnx --img image.jpg --save pose.jpg
```

### Debugging

Use `--debug` to peek at the raw model output and confirm whether the exported
model already applied sigmoid/normalization. If all numbers lie in `[0, 1]` the
script will skip the extra sigmoid to avoid collapsing keypoints to the same
position.

```bash
python main.py --model ../../onnx/yolov8n-pose-fp32.onnx --debug
```

Example output:

```
Raw model output sample: [0.12, -1.3, ...]
Outputs normalized: False
Decoded sample anchors:
{'box': (42.1, 53.4, 118.7, 201.2), 'score': 0.84, 'keypoints': [(60.2, 80.1, 0.90), (90.5, 100.7, 0.88), ...]}
```
