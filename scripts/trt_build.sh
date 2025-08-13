#!/usr/bin/env bash
set -e
ONNX="$1"                    # e.g., onnx/yolov8n-pose-fp32.onnx
ENGINE="$2"                  # e.g., onnx/yolov8n-pose-int8.engine
CALIB="$3"                   # directory with images for calibration

trtexec \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --explicitBatch \
  --int8 --fp16 \
  --calib=Entropy_2 \
  --calibImages="$CALIB" \
  --workspace=4096 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:4x3x640x640 
