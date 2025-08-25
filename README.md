# Quantization for YOLOv8-Pose & TrackNet1000

本專案提供 `YOLOv8-Pose` 與 `TrackNet1000`（基於 YOLOv8 的單球追蹤器）的 ONNX 匯出與 ONNX Runtime INT8 後訓練量化，亦可選擇進行僅 head 的 QAT 或使用 TensorRT。
## 執行步驟

### 0. 設定 Python Path（必須先執行）
在執行任何以下指令前，請先將專案根目錄加入 `PYTHONPATH`，以確保腳本能正確匯入 `ultralytics` 模組：

```bash
export PYTHONPATH="$PWD/ultralytics:$PYTHONPATH"
```

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

#### 評估結果解讀
- `SPEED`：顯示 FP32 與 INT8 模型的平均推論時間與速度提升倍率。
- `CONSISTENCY`：`mean relative L2 diff` 表示兩模型輸出差異，數值越小越好。參考範圍：
  - < 1%：幾乎無損
  - 1%–2%：精度損失極小
  - 2%–5%：需視任務需求評估
  - > 5%：建議調整量化策略或進行 QAT

當 `mean relative L2 diff` 高於可接受範圍或任務精度顯著下降時，建議執行下列 QAT 流程。

### 7. 精度掉太多時，可嘗試 Head-only QAT
```bash
python scripts/qat_head_minimal.py --pt models/yolov8n-pose.pt --include kpt dfl head --epochs 5
```
完成 QAT 後請依下列步驟重新產生 INT8 模型：
1. 重新匯出 ONNX（`scripts/export_onnx.py`）
2. 依步驟 4~5 進行校正量化
3. 重新執行步驟 6 以確認速度與輸出一致性

## 其他注意事項
- 預設：權重為每通道 INT8，活化值為每張量 UINT8。
- 數值敏感區域建議保持浮點：關鍵點 head 的最後幾層、TrackNet1000 head（heatmap / argmax / decoder）、DFL / decoder、sigmoid / softmax 與後處理（NMS）。
- 在 `calib/` 放入 500–2000 張具代表性的校正影像。
