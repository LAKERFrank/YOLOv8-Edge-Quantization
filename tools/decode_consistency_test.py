import argparse
import numpy as np
import onnxruntime as ort
from preproc import preproc_1ch_letterbox

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fp32', required=True)
    ap.add_argument('--int8', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--nodes-file', required=True)
    args = ap.parse_args()

    with open(args.nodes_file, 'r', encoding='utf-8') as f:
        nodes = [ln.strip() for ln in f if ln.strip()]
    s0 = ort.InferenceSession(args.fp32, providers=['CPUExecutionProvider'])
    s1 = ort.InferenceSession(args.int8, providers=['CPUExecutionProvider'])
    x = preproc_1ch_letterbox(args.image, 640)
    y0 = s0.run(nodes, {s0.get_inputs()[0].name: x})
    y1 = s1.run(nodes, {s1.get_inputs()[0].name: x})
    for name, a, b in zip(nodes, y0, y1):
        a_dec = sigmoid(a)
        b_dec = sigmoid(b)
        d = np.abs(a_dec - b_dec)
        print(f'{name}: mae={d.mean():.6f} max={d.max():.6f}')
