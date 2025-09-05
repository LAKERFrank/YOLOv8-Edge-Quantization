# INT8 Quantization for 1-Channel YOLOv8-Pose (Gray)

This folder contains a static post-training quantization (PTQ) pipeline for a YOLOv8-Pose model that accepts single-channel (grayscale) input. It follows the workflow described in the specification:

1. **Verify** the PyTorch weights are truly 1-channel.
2. **Export** the model to ONNX (optional if you already have an ONNX model).
3. **Prepare** a grayscale calibration data reader that outputs tensors shaped `(1, 1, H, W)`.
4. **Run** static QDQ quantization with per-channel weights for `Conv` and `MatMul` ops.
5. **Evaluate** the FP32 and INT8 ONNX models for numerical deviation.

## Usage

```bash
# 0) Install dependencies
pip install -r requirements.txt

# 1) Check the model is grayscale
python scripts/00_check_model_is_gray.py

# 2) Export ONNX (skip if already available)
python scripts/01_export_onnx_gray.py

# 3) Quantize (QDQ format)
python scripts/03_quantize_static_ort.py

# (the script attempts to run `preprocess_model` to fuse ops before quantization; if
# your onnxruntime build lacks this utility, it falls back to quantizing the raw model)

# 4) Evaluate FP32 vs INT8
python scripts/04_eval_fp32_vs_int8.py

# 5) Pinpoint first failing layer (optional)
python scripts/05_layerwise_compare.py \
  --onnx-fp32 onnx/yolov8n-pose-gray.fp32.onnx \
  --onnx-int8 onnx/yolov8n-pose-gray.int8.qdq.onnx \
  --data-root data/val_images \
  --stage backbone --probe c2f,p3,p4,p5

# For a detailed report with per-layer stats, use validate_pose_int8.py
# (same arguments as above) and add --report out.json if needed
```

Calibration and validation image paths are listed in `data/calib_list.txt` and `data/val_list.txt` respectively.
