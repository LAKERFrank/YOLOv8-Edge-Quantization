# Quantization for YOLOv8-Pose & TrackNet1000

本專案提供 `YOLOv8-Pose` 與 `TrackNet1000`（基於 YOLOv8 的單球追蹤器）的 ONNX 匯出與 ONNX Runtime INT8 後訓練量化，亦可選擇進行僅 head 的 QAT 或使用 TensorRT。
## 執行步驟

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 放置模型權重
將模型權重檔放到 `models/` 資料夾：
```text
models/yolov8n-pose.pt
models/tracknet1000.pt
```

### 3. 匯出 ONNX
```bash
python scripts/export_onnx.py --cfg configs/yolov8_pose.yaml
python scripts/export_onnx.py --cfg configs/tracknet1000.yaml
```

### 4. 檢查節點（調整排除名單）
```bash
python scripts/list_onnx_nodes.py onnx/yolov8n-pose-fp32.onnx > out/yolo_nodes.txt
python scripts/list_onnx_nodes.py onnx/tracknet1000-fp32.onnx > out/track_nodes.txt
```
請依上述輸出內容，修改 `configs/*.yaml` 中 `nodes_to_exclude_substrings` 的設定。

### 5. PTQ-Static 校正量化
```bash
python scripts/ort_ptq_static.py --cfg configs/yolov8_pose.yaml --onnx-in onnx/yolov8n-pose-fp32.onnx --onnx-out onnx/yolov8n-pose-int8.onnx
python scripts/ort_ptq_static.py --cfg configs/tracknet1000.yaml --onnx-in onnx/tracknet1000-fp32.onnx --onnx-out onnx/tracknet1000-int8.onnx
```

### 6. 評估速度與輸出一致性
```bash
python scripts/eval_compare_fp32_int8.py --cfg configs/yolov8_pose.yaml --fp32 onnx/yolov8n-pose-fp32.onnx --int8 onnx/yolov8n-pose-int8.onnx --valdir val/pose
python scripts/eval_compare_fp32_int8.py --cfg configs/tracknet1000.yaml --fp32 onnx/tracknet1000-fp32.onnx --int8 onnx/tracknet1000-int8.onnx --valdir val/track
```

### 7. 精度掉太多時，可嘗試 Head-only QAT
```bash
python scripts/qat_head_minimal.py --pt models/yolov8n-pose.pt --include kpt dfl head --epochs 5
```
如有需要，重新匯出 ONNX 並再跑一次 PTQ。

## 其他注意事項
- 預設：權重為每通道 INT8，活化值為每張量 UINT8。
- 數值敏感區域建議保持浮點：關鍵點 head 的最後幾層、TrackNet1000 head（heatmap / argmax / decoder）、DFL / decoder、sigmoid / softmax 與後處理（NMS）。
- 在 `calib/` 放入 500–2000 張具代表性的校正影像。
