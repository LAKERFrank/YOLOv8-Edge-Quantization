# TensorRT INT8 for YOLO-Pose (1-channel)

## 1) Install
```bash
pip install -r trt_quant/requirements.txt
# TensorRT/pycuda 依環境安裝（Jetson 或 NVIDIA 提供的 wheels/apt）
```

## 2) Prepare calibration images (1ch)
```bash
python trt_quant/scripts/prepare_calib.py \
  --src /path/to/your/images \
  --dst trt_quant/calib/images \
  --num 300 --imgsz 640 --shuffle
```

## 3) Export TensorRT engine

### INT8
```bash
python trt_quant/scripts/export_trt.py \
  --model /path/to/your_pose_1ch.pt \
  --data trt_quant/calib/calib.yaml \
  --int8 --imgsz 640 --batch 1 --device 0 \
  --dynamic \
  --minshape 1,1,480,640 --optshape 1,1,640,640 --maxshape 1,1,1080,1920 \
  --outdir trt_quant/engine --name pose_int8.engine
```

### (Optional) FP16
```bash
python trt_quant/scripts/export_trt.py \
  --model /path/to/your_pose_1ch.pt \
  --fp16 --imgsz 640 --batch 1 --device 0 \
  --dynamic \
  --minshape 1,1,480,640 --optshape 1,1,640,640 --maxshape 1,1,1080,1920 \
  --outdir trt_quant/engine --name pose_fp16.engine
```

## 4) Verify engine
```bash
python trt_quant/scripts/verify_engine.py --engine trt_quant/engine/pose_int8.engine
# 檢查 INPUT 綁定：shape 應為 (-1, 1, -1, -1) 或 C=1
# 若有 trtexec，會印出層級 precision（預期以 Int8 為主）
```

## 5) Quick inference
```bash
python trt_quant/scripts/predict_trt.py \
  --engine trt_quant/engine/pose_int8.engine \
  --source /path/to/test_images_or_video --imgsz 640 --device 0 --save
```

## 6) Compare FP32(.pt) vs INT8(.engine)
```bash
python trt_quant/scripts/compare_pt_vs_trt.py \
  --pt /path/to/your_pose_1ch.pt \
  --engine trt_quant/engine/pose_int8.engine \
  --images /path/to/test_images --imgsz 640 --device 0
```
