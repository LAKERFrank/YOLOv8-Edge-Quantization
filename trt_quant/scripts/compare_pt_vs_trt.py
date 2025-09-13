#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare FP32 .pt vs INT8 .engine at the result level (boxes & keypoints):
- For each image, match detections by IoU (greedy, simple).
- Compute MAE and MAX on matched boxes (xyxy) and keypoints (x,y).
Note: This is a light check, not a layer-by-layer diff.
"""
import argparse, os, glob, numpy as np
from ultralytics import YOLO

def iou_xyxy(a, b):
    # a,b: [4] (x1,y1,x2,y2)
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def to_xyxy(boxes):
    # Ultralytics r.boxes.xyxy is a tensor
    if boxes is None or len(boxes)==0:
        return np.zeros((0,4), dtype=np.float32)
    return boxes.xyxy.cpu().numpy().astype(np.float32)

def to_kpts(kpts):
    if kpts is None or len(kpts)==0:
        return np.zeros((0,17,2), dtype=np.float32)
    # (n,17,2)
    return kpts.xy.cpu().numpy().astype(np.float32)

def match_by_iou(a_xyxy, b_xyxy, thr=0.3):
    # greedy matching
    if len(a_xyxy)==0 or len(b_xyxy)==0:
        return []
    used_b = set()
    pairs = []
    for i, a in enumerate(a_xyxy):
        best_j, best_iou = -1, -1.0
        for j, b in enumerate(b_xyxy):
            if j in used_b: continue
            iou = iou_xyxy(a, b)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= thr:
            pairs.append((i, best_j))
            used_b.add(best_j)
    return pairs

def mae_max(x, y):
    d = np.abs(x - y).reshape(-1)
    if d.size == 0:
        return np.nan, np.nan
    return float(np.mean(d)), float(np.max(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="path to FP32 .pt")
    ap.add_argument("--engine", required=True, help="path to INT8 .engine")
    ap.add_argument("--images", required=True, help="folder of test images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    args = ap.parse_args()

    imgs = sorted([p for p in glob.glob(os.path.join(args.images, "*")) if os.path.isfile(p)])
    if not imgs: raise SystemExit(f"No images in {args.images}")

    m_pt = YOLO(args.pt)
    m_trt = YOLO(args.engine)

    mae_boxes_all, max_boxes_all = [], []
    mae_kpts_all,  max_kpts_all  = [], []
    skipped = 0

    for p in imgs:
        r_pt = m_pt.predict(source=p, imgsz=args.imgsz, device=args.device,
                            conf=args.conf, save=False, stream=False, verbose=False)[0]
        r_tr = m_trt.predict(source=p, imgsz=args.imgsz, device=args.device,
                             conf=args.conf, save=False, stream=False, verbose=False)[0]

        a_boxes = to_xyxy(r_pt.boxes); b_boxes = to_xyxy(r_tr.boxes)
        a_kpts  = to_kpts(getattr(r_pt, "keypoints", None))
        b_kpts  = to_kpts(getattr(r_tr, "keypoints", None))

        pairs = match_by_iou(a_boxes, b_boxes, thr=args.iou_thr)
        if not pairs:
            skipped += 1
            continue

        # accumulate MAE/MAX for matched pairs
        for ia, ib in pairs:
            mae_b, max_b = mae_max(a_boxes[ia], b_boxes[ib])
            mae_boxes_all.append(mae_b); max_boxes_all.append(max_b)
            if len(a_kpts)>ia and len(b_kpts)>ib:
                mae_k, max_k = mae_max(a_kpts[ia], b_kpts[ib])
                mae_kpts_all.append(mae_k); max_kpts_all.append(max_k)

    def summarize(name, arr_mae, arr_max):
        arr_mae = np.array([x for x in arr_mae if not np.isnan(x)], dtype=np.float32)
        arr_max = np.array([x for x in arr_max if not np.isnan(x)], dtype=np.float32)
        if arr_mae.size == 0:
            print(f"{name}: no matched samples.")
        else:
            print(f"{name}: MAE={arr_mae.mean():.6f}  MAX={arr_max.max():.6f}  (N={arr_mae.size})")

    print("\n[RESULT] FP32 vs INT8 (result-level)")
    summarize("Boxes (xyxy)", mae_boxes_all, max_boxes_all)
    summarize("Keypoints (x,y)", mae_kpts_all, max_kpts_all)
    print(f"Skipped images (no matches): {skipped}")

if __name__ == "__main__":
    main()
