import argparse
from pathlib import Path
from typing import List, Sequence

import cv2

from api_infer import PoseTRTInfer


def load_images(paths: Sequence[str]):
    frames = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        frames.append(frame)
    return frames


def load_images_from_dir(img_dir: str) -> List[str]:
    dir_path = Path(img_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    image_paths = sorted(
        str(p)
        for p in dir_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {img_dir}")
    return image_paths


def chunked(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--img", nargs=3, help="Three image paths")
    group.add_argument("--img-dir", help="Directory containing images to process in batches of 3")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    infer = PoseTRTInfer(args.engine, conf=args.conf, iou=args.iou, imgsz=tuple(args.imgsz))
    if args.img_dir:
        image_paths = load_images_from_dir(args.img_dir)
        batches = chunked(image_paths, 3)
        if len(batches[-1]) < 3:
            print(
                f"[demo] warning: dropping last {len(batches[-1])} image(s) to keep batch=3"
            )
            batches = batches[:-1]
        for batch_idx, batch_paths in enumerate(batches):
            frames3 = load_images(batch_paths)
            results = infer.infer_3in3out(frames3)
            print(f"[batch {batch_idx}] len(results) = {len(results)}")
            for idx, res in enumerate(results):
                kpts = res.keypoints
                kpt_instances = 0
                kpt_xy_shape = None
                kpt_conf_shape = None
                if kpts is not None:
                    kpt_instances = kpts.xy.shape[0]
                    kpt_xy_shape = kpts.xy.shape
                    if kpts.conf is not None:
                        kpt_conf_shape = kpts.conf.shape
                print(
                    f"[{idx}] orig_shape={res.orig_shape}, boxes={res.boxes.shape}, "
                    f"kpt_instances={kpt_instances}"
                )
                if kpt_xy_shape is not None:
                    print(f"keypoints.xy shape: {kpt_xy_shape}")
                if kpt_conf_shape is not None:
                    print(f"keypoints.conf shape: {kpt_conf_shape}")
    else:
        frames3 = load_images(args.img)
        results = infer.infer_3in3out(frames3)
        print(f"len(results) = {len(results)}")
        for idx, res in enumerate(results):
            kpts = res.keypoints
            kpt_instances = 0
            kpt_xy_shape = None
            kpt_conf_shape = None
            if kpts is not None:
                kpt_instances = kpts.xy.shape[0]
                kpt_xy_shape = kpts.xy.shape
                if kpts.conf is not None:
                    kpt_conf_shape = kpts.conf.shape
            print(
                f"[{idx}] orig_shape={res.orig_shape}, boxes={res.boxes.shape}, "
                f"kpt_instances={kpt_instances}"
            )
            if kpt_xy_shape is not None:
                print(f"keypoints.xy shape: {kpt_xy_shape}")
            if kpt_conf_shape is not None:
                print(f"keypoints.conf shape: {kpt_conf_shape}")


if __name__ == "__main__":
    main()
