import argparse, yaml, os, glob, cv2, time, numpy as np
import onnxruntime as ort
from tqdm import tqdm

def preprocess(path, imgsz, lb_val, norm, mean, std):
    im = cv2.imread(path)
    h, w = im.shape[:2]
    r = imgsz / max(h, w)
    nh, nw = int(round(h*r)), int(round(w*r))
    im_res = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), lb_val, dtype=im.dtype)
    top, left = (imgsz-nh)//2, (imgsz-nw)//2
    canvas[top:top+nh, left:left+nw] = im_res
    img = canvas[:, :, ::-1].transpose(2,0,1).astype(np.float32)
    if norm:
        img /= 255.0
        mean_arr = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        std_arr = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
        if mean_arr.shape[0] != img.shape[0]:
            mean_arr = np.resize(mean_arr, (img.shape[0], 1, 1))
        if std_arr.shape[0] != img.shape[0]:
            std_arr = np.resize(std_arr, (img.shape[0], 1, 1))
        img = (img - mean_arr) / std_arr
    return img

def run_session(onnx_path, input_name=None):
    so = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = input_name or sess.get_inputs()[0].name
    return sess, input_name

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--valdir", default="val")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    imgs = sorted(glob.glob(os.path.join(args.valdir, "*.*")))[:args.limit]
    if not imgs: raise SystemExit("No val images at " + args.valdir)

    fp_sess, in_name = run_session(args.fp32, cfg.get("input_name","images"))
    int8_sess, _ = run_session(args.int8, in_name)

    diffs, t_fp, t_i8 = [], [], []
    for p in tqdm(imgs, desc="Eval"):
        x = preprocess(p, cfg["imgsz"], cfg["letterbox_value"], cfg["normalize"], cfg["mean"], cfg["std"])
        x = np.expand_dims(x, 0)
        t0 = time.time(); y_fp = fp_sess.run(None, {in_name: x}); t_fp.append(time.time()-t0)
        t0 = time.time(); y_i8 = int8_sess.run(None, {in_name: x}); t_i8.append(time.time()-t0)
        yfp = np.concatenate([o.reshape(-1).astype(np.float32) for o in y_fp])
        yi8 = np.concatenate([o.reshape(-1).astype(np.float32) for o in y_i8])
        n = min(len(yfp), len(yi8))
        diffs.append(np.linalg.norm(yfp[:n]-yi8[:n]) / (np.linalg.norm(yfp[:n])+1e-6))

    print(f"[SPEED] FP32 avg {np.mean(t_fp):.4f}s | INT8 avg {np.mean(t_i8):.4f}s | speedup x{np.mean(t_fp)/max(np.mean(t_i8),1e-6):.2f}")
    print(f"[CONSISTENCY] mean relative L2 diff: {np.mean(diffs):.6f}")
    # TODO: add OKS/PCK (pose) and pixel MAE/hit@r (track) if labels are available.
