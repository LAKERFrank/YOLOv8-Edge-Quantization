# Quantization for YOLOv8-Pose & TrackNet1000

This repo provides ONNX export + ONNX Runtime INT8 PTQ (and optional head-only QAT / TensorRT) for:
- `YOLOv8-Pose`
- `TrackNet1000` (YOLOv8-based single-shuttle tracker)

## Quick Start
1) `pip install -r requirements.txt`
2) Put your weights:
   - `models/yolov8n-pose.pt`
   - `models/tracknet1000.pt`
3) Export ONNX:
python scripts/export_onnx.py --cfg configs/yolov8_pose.yaml
python scripts/export_onnx.py --cfg configs/tracknet1000.yaml

css
複製
編輯
4) Inspect nodes (to refine exclusions):
python scripts/list_onnx_nodes.py onnx/yolov8n-pose-fp32.onnx > out/yolo_nodes.txt
python scripts/list_onnx_nodes.py onnx/tracknet1000-fp32.onnx > out/track_nodes.txt

lua
複製
編輯
Edit `nodes_to_exclude_substrings` in the config files to match your actual node names.
5) PTQ-Static (calibrated INT8):
python scripts/ort_ptq_static.py --cfg configs/yolov8_pose.yaml
--onnx-in onnx/yolov8n-pose-fp32.onnx --onnx-out onnx/yolov8n-pose-int8.onnx

python scripts/ort_ptq_static.py --cfg configs/tracknet1000.yaml
--onnx-in onnx/tracknet1000-fp32.onnx --onnx-out onnx/tracknet1000-int8.onnx

bash
複製
編輯
6) Quick eval (speed + output consistency):
python scripts/eval_compare_fp32_int8.py --cfg configs/yolov8_pose.yaml
--fp32 onnx/yolov8n-pose-fp32.onnx --int8 onnx/yolov8n-pose-int8.onnx --valdir val/pose

python scripts/eval_compare_fp32_int8.py --cfg configs/tracknet1000.yaml
--fp32 onnx/tracknet1000-fp32.onnx --int8 onnx/tracknet1000-int8.onnx --valdir val/track

pgsql
複製
編輯
7) If accuracy drop is high, try **head-only QAT**:
python scripts/qat_head_minimal.py --pt models/yolov8n-pose.pt --include kpt dfl head --epochs 5

bash
複製
編輯
Re-export ONNX and re-run PTQ if needed.

## Notes
- Default: INT8 per-channel weights, per-tensor UINT8 activations.
- Keep numerically sensitive parts in float: the very last layers of keypoint head, TrackNet1000 head (heatmap/argmax/decoder), DFL/decoder, sigmoid/softmax, and post-processing (NMS).
- Use 500–2000 **representative** calibration images in `calib/`.
