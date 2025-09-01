import argparse
import json
import os
import random
from glob import glob
from typing import Dict, List

import cv2
import numpy as np
import onnx
import onnxruntime as ort


# -----------------------
# Preprocessing utilities
# -----------------------

def letterbox_gray(img: np.ndarray, new_shape=640, pad=114, mode='ultra'):
    """Resize keeping aspect ratio, pad to square."""
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if mode == 'ultra':
        dw /= 2
        dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad)
    return img, r, (dw, dh)


def preprocess_gray_1ch(path: str, img_size: int, letterbox_mode='ultra',
                        scale=1/255.0, mean=0.0, std=1.0, layout='NCHW'):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f'Failed to read image: {path}'
    img, _, _ = letterbox_gray(img, new_shape=img_size, pad=114, mode=letterbox_mode)
    img = img.astype('float32') * scale
    if std != 1.0 or mean != 0.0:
        img = (img - mean) / std
    if layout.upper() == 'NCHW':
        img = img[None, None, ...]
    else:  # NHWC
        img = img[..., None][None, ...]
    return img


# -----------------------
# ONNX helpers
# -----------------------

def find_probe_names(model: onnx.ModelProto, patterns: List[str]) -> List[str]:
    names = []
    for node in model.graph.node:
        for out in node.output:
            for p in patterns:
                if p and p in out:
                    names.append(out)
    return list(dict.fromkeys(names))


def add_model_outputs(model_path: str, extra: List[str]) -> str:
    if not extra:
        return model_path
    model = onnx.load(model_path)
    existing = {o.name for o in model.graph.output}
    for name in extra:
        if name in existing:
            continue
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    tmp_path = model_path + ".tmp.out.onnx"
    onnx.save(model, tmp_path)
    return tmp_path


def run_sess(sess: ort.InferenceSession, x: np.ndarray) -> Dict[str, np.ndarray]:
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, {sess.get_inputs()[0].name: x})
    return dict(zip(names, outs))


# -----------------------
# Stats helpers
# -----------------------

def init_stats():
    return {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'sumsq': 0.0, 'n': 0, 'shape': None}


def update_stats(stats: dict, arr: np.ndarray):
    stats['min'] = min(stats['min'], float(arr.min()))
    stats['max'] = max(stats['max'], float(arr.max()))
    stats['sum'] += float(arr.sum())
    stats['sumsq'] += float((arr ** 2).sum())
    stats['n'] += arr.size
    stats['shape'] = list(arr.shape)


def finalize_stats(stats: dict):
    mean = stats['sum'] / stats['n'] if stats['n'] else 0.0
    var = stats['sumsq'] / stats['n'] - mean ** 2 if stats['n'] else 0.0
    var = max(var, 0.0)
    return {'min': stats['min'], 'max': stats['max'], 'mean': mean, 'std': var ** 0.5}


# -----------------------
# Main evaluation
# -----------------------

def main(args):
    img_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        img_paths.extend(glob(os.path.join(args.data_root, ext)))
    assert img_paths, f'No images found under {args.data_root}'
    random.seed(args.seed)
    random.shuffle(img_paths)
    img_paths = img_paths[:args.num_samples]

    # Parse normalization
    scale = args.scale
    mean = args.mean
    std = args.std

    # Prepare probe names
    probe_patterns = [p.strip() for p in args.probe.split(',') if p.strip()] if args.probe else []
    model_fp32 = onnx.load(args.onnx_fp32)
    probe_names = find_probe_names(model_fp32, probe_patterns)
    # add final outputs to ensure they are included
    probe_names += [o.name for o in model_fp32.graph.output]
    probe_names = list(dict.fromkeys(probe_names))

    fp32_model_path = add_model_outputs(args.onnx_fp32, probe_names)
    int8_model_path = add_model_outputs(args.onnx_int8, probe_names)

    providers = ['CUDAExecutionProvider'] if args.ort_provider.lower() == 'cuda' else ['CPUExecutionProvider']
    sess_fp32 = ort.InferenceSession(fp32_model_path, providers=providers)
    sess_int8 = ort.InferenceSession(int8_model_path, providers=providers)
    out_names = [o.name for o in sess_fp32.get_outputs()]

    metrics = {n: {'mae_sum': 0.0, 'max': 0.0, 'count': 0} for n in out_names}
    stats_a = {n: init_stats() for n in out_names}
    stats_b = {n: init_stats() for n in out_names}
    preproc_stats = init_stats()

    for path in img_paths:
        x = preprocess_gray_1ch(path, args.img_size, args.letterbox, scale, mean, std, args.layout)
        update_stats(preproc_stats, x)
        outputs_fp32 = run_sess(sess_fp32, x)
        outputs_int8 = run_sess(sess_int8, x)
        for name in out_names:
            a = outputs_fp32[name]
            b = outputs_int8[name]
            diff = np.abs(a - b)
            metrics[name]['mae_sum'] += float(diff.mean())
            metrics[name]['max'] = max(metrics[name]['max'], float(diff.max()))
            metrics[name]['count'] += 1
            update_stats(stats_a[name], a)
            update_stats(stats_b[name], b)

    per_layer = [
        {
            'name': 'preproc',
            'mae': 0.0,
            'max': 0.0,
            'shape': preproc_stats['shape'],
            'stats_fp32': finalize_stats(preproc_stats),
            'stats_int8': finalize_stats(preproc_stats),
        }
    ]
    first_failure = None
    for name in out_names:
        m = metrics[name]
        mae = m['mae_sum'] / m['count'] if m['count'] else 0.0
        mx = m['max']
        layer_info = {
            'name': name,
            'mae': mae,
            'max': mx,
            'shape': stats_a[name]['shape'],
            'stats_fp32': finalize_stats(stats_a[name]),
            'stats_int8': finalize_stats(stats_b[name]),
        }
        per_layer.append(layer_info)
        if first_failure is None and (mae > args.threshold_mae or mx > args.threshold_max):
            first_failure = {'layer': name, 'mae': mae, 'max': mx}

    summary = per_layer[-1] if per_layer else {}
    print(f"[Summary] layer={summary.get('name','')} MAE={summary.get('mae',0):.6f} MAX={summary.get('max',0):.6f}")
    for pl in per_layer:
        print(f"{pl['name']}: MAE={pl['mae']:.6f} MAX={pl['max']:.6f}")
    if first_failure:
        print("first_failure:", first_failure)
    else:
        print("All layers within thresholds")

    report = {
        'summary': summary,
        'per_layer': per_layer,
        'first_failure': first_failure,
        'config': vars(args),
    }
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

    if args.dump_dir and first_failure:
        os.makedirs(args.dump_dir, exist_ok=True)
        # dump outputs for all failing layers using first image
        x = preprocess_gray_1ch(img_paths[0], args.img_size, args.letterbox, scale, mean, std, args.layout)
        outs_a = run_sess(sess_fp32, x)
        outs_b = run_sess(sess_int8, x)
        for name, m in metrics.items():
            mae = m['mae_sum']/m['count'] if m['count'] else 0.0
            mx = m['max']
            if mae > args.threshold_mae or mx > args.threshold_max:
                np.save(os.path.join(args.dump_dir, f'{name}_fp32.npy'), outs_a[name])
                np.save(os.path.join(args.dump_dir, f'{name}_int8.npy'), outs_b[name])
                with open(os.path.join(args.dump_dir, f'{name}_stats.txt'), 'w') as f:
                    f.write(json.dumps(next(pl for pl in per_layer if pl['name']==name), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate FP32 vs INT8 outputs for 1ch pose model')
    parser.add_argument('--onnx-fp32', required=True, help='FP32 ONNX path')
    parser.add_argument('--onnx-int8', required=True, help='INT8 ONNX path')
    parser.add_argument('--data-root', required=True, help='Directory of validation images')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--layout', choices=['NCHW', 'NHWC'], default='NCHW')
    parser.add_argument('--letterbox', default='ultra')
    parser.add_argument('--normalize', default='scale=1/255,mean=0.0,std=1.0')
    parser.add_argument('--stage', default='head_raw')
    parser.add_argument('--probe', default='')
    parser.add_argument('--threshold-mae', type=float, default=0.02)
    parser.add_argument('--threshold-max', type=float, default=0.1)
    parser.add_argument('--ort-provider', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dump-dir', default='')
    parser.add_argument('--report', default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # parse normalize string
    norm_parts = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in args.normalize.split(',') if '=' in kv}
    args.scale = norm_parts.get('scale', 1/255.0)
    args.mean = norm_parts.get('mean', 0.0)
    args.std = norm_parts.get('std', 1.0)

    main(args)
