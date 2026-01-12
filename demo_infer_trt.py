import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

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


def _draw_pose(
    image: np.ndarray,
    boxes: np.ndarray,
    keypoints: np.ndarray,
    kpt_conf: np.ndarray | None,
    conf_thr: float,
) -> np.ndarray:
    skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    output = image.copy()
    for box in boxes.astype(int):
        cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    if keypoints.size == 0:
        return output
    for person_idx, person_kpts in enumerate(keypoints):
        for kpt_idx, (x, y) in enumerate(person_kpts):
            if kpt_conf is not None and kpt_conf[person_idx, kpt_idx] < conf_thr:
                continue
            cv2.circle(output, (int(x), int(y)), 3, (0, 255, 255), -1)
        for a, b in skeleton:
            if kpt_conf is not None:
                if kpt_conf[person_idx, a] < conf_thr or kpt_conf[person_idx, b] < conf_thr:
                    continue
            x1, y1 = person_kpts[a]
            x2, y2 = person_kpts[b]
            cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    return output


def _save_outputs(
    out_dir: Path,
    batch_paths: List[str],
    frames: Sequence[np.ndarray],
    results,
    conf_thr: float,
    batch_idx: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (path, frame, res) in enumerate(zip(batch_paths, frames, results)):
        stem = Path(path).stem
        if res.keypoints is not None:
            kpts = res.keypoints.xy
            kpt_conf = res.keypoints.conf
        else:
            kpts = np.zeros((0, 17, 2), dtype=np.float32)
            kpt_conf = None
        drawn = _draw_pose(frame, res.boxes, kpts, kpt_conf, conf_thr)
        out_path = out_dir / f"batch{batch_idx:03d}_{idx:02d}_{stem}.jpg"
        cv2.imwrite(str(out_path), drawn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--img", nargs=3, help="Three image paths")
    group.add_argument("--img-dir", help="Directory containing images to process in batches of 3")
    parser.add_argument("--out-dir", default="trt_outputs", help="Directory to save drawn results")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    infer = PoseTRTInfer(args.engine, conf=args.conf, iou=args.iou, imgsz=tuple(args.imgsz))
    out_dir = Path(args.out_dir)
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
            _save_outputs(out_dir, batch_paths, frames3, results, args.conf, batch_idx)
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
        _save_outputs(out_dir, list(args.img), frames3, results, args.conf, 0)
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
