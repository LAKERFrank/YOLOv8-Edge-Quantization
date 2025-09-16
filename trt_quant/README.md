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

### INT8（Ultralytics export path）
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

### Alternative: build INT8 from ONNX with custom calibration
如果你已經有 1-channel 的 ONNX（例如透過 `export_trt.py --onnx-only` 產生），可以使用
`trt_quant/scripts/build_int8.py` 直接建出兩顆 INT8 engine，並對關鍵層啟用 FP16 fallback。

校正資料夾需放入「偏黑背景 + 後場遠距人物」的代表性影像（建議 1000–3000 張），下面以
`trt_quant/calib/dark_small/` 為例：

```bash
# MinMax 版本（預設）
python trt_quant/scripts/build_int8.py \
  --onnx trt_quant/engine/pose_1ch.onnx \
  --out trt_quant/engine/pose_int8_minmax.engine \
  --calib-dir trt_quant/calib/dark_small \
  --imgsz 640 \
  --fp16-fallback "model.0,detect"

# Entropy 版本（A/B 比較）
python trt_quant/scripts/build_int8.py \
  --onnx trt_quant/engine/pose_1ch.onnx \
  --out trt_quant/engine/pose_int8_entropy.engine \
  --calib-dir trt_quant/calib/dark_small \
  --imgsz 640 \
  --calibrator entropy \
  --fp16-fallback "model.0,detect"
```

常用參數：

- `--calib-dir`：校正資料夾，僅讀取 jpg/png 影像，會自動轉灰階 + letterbox 為 1×1×H×W。
- `--fp16-fallback`：以逗號分隔的層名稱關鍵字，預設固定第一層（`model.0`）與 head（`detect`）
  使用 FP16，以穩定暗場輸出。若要全部 INT8，可設為空字串。
- `--enable-fp16 / --disable-fp16`：控制是否允許 TensorRT 使用 FP16 kernel。
- 腳本會列印 network layer 清單，若需要微調 fallback 層，可先查看名稱再調整關鍵字。
- 校正 cache 會以 `*.calib.cache` 存放在 engine 同路徑，下次重複使用會加速 build。

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
