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
  --src-labels /path/to/your/labels \  # optional
  --dst-labels trt_quant/calib/labels \
  --num 300 --imgsz 640 --shuffle
```
`calib.yaml` sets `path: trt_quant/calib`, so run these commands from the repository root.

If `--src-labels` is omitted, blank label files are created in `--dst-labels` to avoid warnings.

Example `trt_quant/calib/calib.yaml`:

```yaml
# 代表性資料（校正用）。影像需為單通道灰階（png/jpg）
path: trt_quant/calib
train: images
val: images
names:
  0: person
kpt_shape: [17, 3]
channels: 1
```

## 3) Export TensorRT engine

Export flow: `.pt -> .onnx -> trtexec build`. ONNX and engine files are saved to `--outdir`. Requires TensorRT 10.7 (container 24.12) with `trtexec` available.

### INT8
```bash
python trt_quant/scripts/export_trt.py \
  --model /path/to/your_pose_1ch.pt \
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

Shape flags are only required when `--dynamic` is set; omit them for static models.

## 4) Verify engine
```bash
python trt_quant/scripts/verify_engine.py --engine trt_quant/engine/pose_int8.engine
# 檢查 INPUT 綁定：shape 應為 (-1, 1, -1, -1) 或 C=1
# 若有 trtexec，會印出層級 precision（預期以 Int8 為主）
```

## 5) Quick inference
`predict_trt.py` runs the TensorRT engine directly (no Ultralytics
pipeline) on an image, video, or a directory of images. Bounding boxes
are drawn in blue with confidence labels and keypoints in `#f5ac36`.
Annotated results are saved to `runs/predict`. Sources are converted to
grayscale automatically when the engine expects a single-channel tensor.
If your engine omits class probabilities (e.g. a single-class model),
the script adjusts automatically; otherwise set `--nc` to match your
class count.

Optional low-light enhancement is available with `--ll-enhance`.
Gamma, CLAHE usage, clip limit, and grid size can be tuned via
`--ll-gamma`, `--ll-clahe/--no-ll-clahe`, `--ll-clip`, and `--ll-grid`
respectively.

```bash
python trt_quant/scripts/predict_trt.py \
  --engine trt_quant/engine/pose_int8.engine \
  --source /path/to/test.jpg \
  --imgsz 640 --device 0 --save
```

Replace `test.jpg` with a video file or a directory of images to process
multiple frames. Add `--show` to display the output in a window while
running.

## 6) Compare FP32(.pt) vs INT8(.engine)
```bash
python trt_quant/scripts/compare_pt_vs_trt.py \
  --pt /path/to/your_pose_1ch.pt \
  --engine trt_quant/engine/pose_int8.engine \
  --images /path/to/test_images --imgsz 640 --device 0
```
