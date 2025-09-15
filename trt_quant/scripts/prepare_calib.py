#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare grayscale (1ch) calibration images for TensorRT INT8.
"""
import argparse, os, random, shutil, glob
import cv2

def is_image(p):
    ext = os.path.splitext(p)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def resize_with_padding(img, new_size, color=114):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_w, pad_h = new_size - new_w, new_size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source folder of images")
    ap.add_argument("--dst", default="trt_quant/calib/images", help="dest folder")
    ap.add_argument("--src-labels", help="optional source folder of labels")
    ap.add_argument("--dst-labels", default="trt_quant/calib/labels",
                    help="dest folder for labels")
    ap.add_argument("--num", type=int, default=300, help="number of images")
    ap.add_argument("--imgsz", type=int, default=640, help="resize to square (0=keep)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle selection")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    os.makedirs(args.dst_labels, exist_ok=True)
    files = [p for p in glob.glob(os.path.join(args.src, "**"), recursive=True) if is_image(p)]
    if not files:
        raise SystemExit(f"No images found in {args.src}")

    if args.shuffle:
        random.shuffle(files)
    else:
        files.sort()

    picked = files[:args.num]
    for i, p in enumerate(picked, 1):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"[skip] cannot read: {p}")
            continue
        if args.imgsz > 0:
            g = resize_with_padding(g, args.imgsz)
        out = os.path.join(args.dst, f"calib_{i:04d}.png")
        cv2.imwrite(out, g)

        label_out = os.path.join(args.dst_labels, f"calib_{i:04d}.txt")
        if args.src_labels:
            rel = os.path.relpath(p, args.src)
            label_in = os.path.join(
                args.src_labels, os.path.splitext(rel)[0] + ".txt")
            if os.path.isfile(label_in):
                shutil.copy(label_in, label_out)
            else:
                open(label_out, "w").close()
        else:
            open(label_out, "w").close()
    print(f"Done. Wrote {len(picked)} grayscale images to: {args.dst}")

if __name__ == "__main__":
    main()
