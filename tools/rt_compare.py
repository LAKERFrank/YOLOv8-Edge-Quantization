import argparse
import csv
import glob
import json
import numpy as np
import onnxruntime as ort

def run_sess(sess, x, names):
    inp = {sess.get_inputs()[0].name: x}
    return sess.run(names, inp)

def mae_max(a, b):
    d = np.abs(a - b)
    return float(d.mean()), float(d.max(initial=0))

def collect_range(arr):
    return float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--fp32', required=True)
    ap.add_argument('--int8', required=True)
    ap.add_argument('--images', nargs='+', required=False)
    ap.add_argument('--compare-point', choices=['predecode', 'postdecode'], default='postdecode')
    ap.add_argument('--nodes-file')
    ap.add_argument('--save-csv')
    ap.add_argument('--log-range')
    ap.add_argument('--synthetic', action='store_true')
    args = ap.parse_args()

    s_fp32 = ort.InferenceSession(args.fp32, providers=['CPUExecutionProvider'])
    s_int8 = ort.InferenceSession(args.int8, providers=['CPUExecutionProvider'])

    if args.compare-point == 'postdecode':
        outs = [o.name for o in s_fp32.get_outputs()]
    else:
        assert args.nodes_file, '--nodes-file required for predecode'
        with open(args.nodes_file, 'r', encoding='utf-8') as f:
            outs = [ln.strip() for ln in f if ln.strip()]

    rows = []
    images = []
    if args.synthetic:
        images = ['zeros', 'half', 'uniform']
    elif args.images:
        for pattern in args.images:
            images.extend(sorted(glob.glob(pattern)))
    else:
        raise SystemExit('No images provided')

    for p in images:
        if p == 'zeros':
            x = np.zeros((1,1,640,640), dtype=np.float32)
        elif p == 'half':
            x = np.full((1,1,640,640), 0.5, dtype=np.float32)
        elif p == 'uniform':
            x = np.random.rand(1,1,640,640).astype(np.float32)
        else:
            from preproc import preproc_1ch_letterbox
            x = preproc_1ch_letterbox(p, 640)
        y0 = run_sess(s_fp32, x, outs)
        y1 = run_sess(s_int8, x, outs)
        for name, a, b in zip(outs, y0, y1):
            m, M = mae_max(a, b)
            r = {'image': p, 'node': name, 'mae': m, 'maxabs': M}
            if args.log-range:
                rmin, rmax, rmean, rstd = collect_range(a)
                r['min'] = rmin; r['max'] = rmax
                r['mean'] = rmean; r['std'] = rstd
            rows.append(r)
    if args.save_csv:
        with open(args.save_csv, 'w', newline='') as f:
            fieldnames = list(rows[0].keys()) if rows else ['image','node','mae','maxabs']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
    worst = sorted(rows, key=lambda r: r['mae'], reverse=True)[:10]
    for r in worst:
        print(f"{r['node']}: MAE={r['mae']:.6f} MAX={r['maxabs']:.6f}")
    if args.log-range:
        with open(args.log-range, 'w', newline='') as f:
            fieldnames = list(rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
