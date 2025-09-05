import argparse
import os
import random
import sys
from glob import glob

import cv2  # noqa: F401  (imported for preprocessing)
import numpy as np
import onnx
import onnxruntime as ort

# Reuse helpers from validate_pose_int8.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from validate_pose_int8 import (  # type: ignore
    STAGE_PROBES,
    preprocess_gray_1ch,
    find_probe_names,
    add_model_outputs,
    run_sess,
    infer_stage,
)


def _parse_num(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        if "/" in v:
            a, b = v.split("/", 1)
            return float(a) / float(b)
        raise


def parse_normalize(s: str):
    parts = {}
    for kv in s.split(','):
        if '=' in kv:
            k, v = kv.split('=', 1)
            parts[k] = _parse_num(v)
    return (
        parts.get('scale', 1/255.0),
        parts.get('mean', 0.0),
        parts.get('std', 1.0),
    )


def main(args):
    img_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        img_paths.extend(glob(os.path.join(args.data_root, ext)))
    assert img_paths, f"No images found under {args.data_root}"
    random.seed(args.seed)
    random.shuffle(img_paths)
    img_paths = img_paths[:args.num_samples]

    scale, mean, std = parse_normalize(args.normalize)

    probe_patterns = list(STAGE_PROBES.get(args.stage, []))
    if args.probe:
        probe_patterns.extend([p.strip() for p in args.probe.split(',') if p.strip()])

    model_fp32 = onnx.load(args.onnx_fp32)
    names = find_probe_names(model_fp32, probe_patterns)
    names += [o.name for o in model_fp32.graph.output]
    names = list(dict.fromkeys(names))

    fp32_path = add_model_outputs(args.onnx_fp32, names)
    int8_path = add_model_outputs(args.onnx_int8, names)

    providers = ['CUDAExecutionProvider'] if args.provider == 'cuda' else ['CPUExecutionProvider']
    sess_fp32 = ort.InferenceSession(fp32_path, providers=providers)
    sess_int8 = ort.InferenceSession(int8_path, providers=providers)

    out_names = [o.name for o in sess_fp32.get_outputs()]
    metrics = {n: {'mae_sum': 0.0, 'max': 0.0, 'count': 0} for n in out_names}

    for path in img_paths:
        x = preprocess_gray_1ch(path, args.img_size, args.letterbox, scale, mean, std, args.layout)
        outs_a = run_sess(sess_fp32, x)
        outs_b = run_sess(sess_int8, x)
        for n in out_names:
            a = outs_a[n]
            b = outs_b[n]
            if a.shape != b.shape:
                raise RuntimeError(f"shape mismatch for {n}: {a.shape} vs {b.shape}")
            d = np.abs(a - b)
            metrics[n]['mae_sum'] += float(d.mean())
            metrics[n]['max'] = max(metrics[n]['max'], float(d.max()))
            metrics[n]['count'] += 1

    first_failure = None
    for n in out_names:
        m = metrics[n]
        mae = m['mae_sum'] / m['count'] if m['count'] else 0.0
        mx = m['max']
        print(f"{n}: MAE={mae:.6f} MAX={mx:.6f}")
        if first_failure is None and (mae > args.threshold_mae or mx > args.threshold_max):
            first_failure = {'layer': n, 'stage': infer_stage(n), 'mae': mae, 'max': mx}

    if first_failure:
        print('first_failure:', first_failure)
    else:
        print('All layers within thresholds')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Layer-wise FP32 vs INT8 comparison')
    ap.add_argument('--onnx-fp32', required=True)
    ap.add_argument('--onnx-int8', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--img-size', type=int, default=640)
    ap.add_argument('--num-samples', type=int, default=200)
    ap.add_argument('--layout', choices=['NCHW', 'NHWC'], default='NCHW')
    ap.add_argument('--letterbox', default='ultra')
    ap.add_argument('--normalize', default='scale=1/255,mean=0.0,std=1.0')
    ap.add_argument('--stage', default='head_raw')
    ap.add_argument('--probe', default='')
    ap.add_argument('--threshold-mae', type=float, default=0.02)
    ap.add_argument('--threshold-max', type=float, default=0.1)
    ap.add_argument('--provider', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    main(args)

