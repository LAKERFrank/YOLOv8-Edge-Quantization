# YOLOv8-Edge-Quantization

Quantization for YOLOv8 Pose & TrackNet1000.

## TensorRT YOLOv8 Pose 3-in/3-out Manual

This repo includes a TensorRT inference pipeline that accepts **three input frames** and returns a list of **three results** in the same format as the PyTorch version.

### Stage 1: Prepare assets

1. **TensorRT engine**: export a YOLOv8 Pose engine that supports explicit batch and dynamic shape with an optimization profile where `max` batch is at least 3.
2. **PyTorch weights** (optional, for validation): `*.pt` file for the same model.
3. **Inputs**: three images or three frames extracted from a video.

### Stage 2: Run demo inference (TensorRT only)

This stage validates the TensorRT pipeline and prints the required summary to confirm the `3-in/3-out` behavior.

```bash
python demo_infer_trt.py \
  --engine path/to/model.engine \
  --img path/to/img1.jpg path/to/img2.jpg path/to/img3.jpg \
  --imgsz 640 640 \
  --conf 0.25 \
  --iou 0.7
```

To process an entire folder in batches of 3 images:

```bash
python demo_infer_trt.py \
  --engine path/to/model.engine \
  --img-dir path/to/images \
  --imgsz 640 640 \
  --conf 0.25 \
  --iou 0.7
```

Expected output includes:

- `len(results) = 3`
- per-frame summary with `orig_shape`, `boxes`, and keypoint counts
- `keypoints.xy shape: (N, 17, 2)` (and `keypoints.conf shape` if available)

### Stage 3: Validate vs PyTorch (optional but recommended)

This stage compares TensorRT output against PyTorch on the same 3-frame input to catch preprocess or decode mismatches quickly.

```bash
python validate_vs_torch.py \
  --engine path/to/model.engine \
  --weights path/to/model.pt \
  --img path/to/img1.jpg path/to/img2.jpg path/to/img3.jpg \
  --imgsz 640 640 \
  --conf 0.25 \
  --iou 0.7
```

To validate a folder in batches of 3 images:

```bash
python validate_vs_torch.py \
  --engine path/to/model.engine \
  --weights path/to/model.pt \
  --img-dir path/to/images \
  --imgsz 640 640 \
  --conf 0.25 \
  --iou 0.7
```

The script prints:

- number of boxes per frame (PyTorch vs TRT)
- mean keypoint pixel error for matched detections
- mean confidence for matched boxes

### Stage 4: Use the API in your own code

```python
from api_infer import PoseTRTInfer

infer = PoseTRTInfer(
    engine_path="path/to/model.engine",
    conf=0.25,
    iou=0.7,
    imgsz=(640, 640),
)

results = infer.infer_3in3out([frame1, frame2, frame3])
print(len(results))  # 3
```

### Stage 5: Debug helpers

- Inspect engine bindings:

```python
from trt_runner import TrtRunner

runner = TrtRunner("path/to/model.engine")
runner.dump_bindings()
```

- Enable verbose shape prints during inference:

```python
outputs = runner.infer(batch, verbose=True)
```
