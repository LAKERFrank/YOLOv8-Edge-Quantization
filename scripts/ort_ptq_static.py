import argparse, os, glob, yaml, cv2, numpy as np, onnx
from tqdm import tqdm
from onnxruntime.quantization import (
    CalibrationDataReader, quantize_static, QuantType, CalibrationMethod
)

def letterbox(im, new=640, color=114):
    h, w = im.shape[:2]
    r = new / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    if im.ndim == 2:
        canvas = np.full((new, new), color, dtype=im.dtype)
    else:
        canvas = np.full((new, new, im.shape[2]), color, dtype=im.dtype)
    canvas[top:top + nh, left:left + nw] = im
    return canvas

class ImageCalibReader(CalibrationDataReader):
    def __init__(self, img_dir, input_name, size, norm, mean, std):
        self.files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.i = 0
        self.input_name = input_name
        self.size = size
        self.norm = norm
        self.c = len(mean)
        self.mean = np.array(mean).reshape(self.c, 1, 1)
        self.std = np.array(std).reshape(self.c, 1, 1)
        self.group = 1 if self.c in (1, 3) else self.c

    def get_next(self):
        if self.i >= len(self.files):
            return None

        if self.c == 3:
            img = cv2.imread(self.files[self.i])
            self.i += 1
            img = letterbox(img, self.size)
            img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        elif self.c == 1:
            img = cv2.imread(self.files[self.i], cv2.IMREAD_GRAYSCALE)
            self.i += 1
            img = letterbox(img, self.size)
            img = np.expand_dims(img, 0).astype(np.float32)
        else:  # stacked grayscale
            if self.i + self.group > len(self.files):
                return None
            imgs = []
            for _ in range(self.group):
                im = cv2.imread(self.files[self.i], cv2.IMREAD_GRAYSCALE)
                self.i += 1
                im = letterbox(im, self.size)
                imgs.append(im)
            img = np.stack(imgs, axis=0).astype(np.float32)

        if self.norm:
            img /= 255.0
            img = (img - self.mean) / self.std

        return {self.input_name: np.expand_dims(img, 0)}

def build_exclude_list(onnx_path, substrings):
    m = onnx.load(onnx_path)
    ex = []
    for n in m.graph.node:
        fullname = (n.name or "") + "|" + ",".join(n.output)
        if any(s in fullname for s in substrings):
            ex.append(n.name)
    return list(sorted(set(ex)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--onnx-in", required=True)
    ap.add_argument("--onnx-out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    input_name = cfg.get("input_name","images")
    size = int(cfg.get("imgsz",640))
    dr = ImageCalibReader(
        img_dir=cfg["calibration_images_dir"], input_name=input_name, size=size,
        norm=bool(cfg.get("normalize", True)), mean=cfg.get("mean",[0,0,0]), std=cfg.get("std",[1,1,1])
    )

    act_dt = QuantType.QUInt8 if cfg.get("activation_dtype","uint8")=="uint8" else QuantType.QInt8
    wt_dt  = QuantType.QInt8

    method = cfg.get("calibration_method","percentile_999")
    if method == "minmax":
        cali = CalibrationMethod.MinMax
    elif method == "entropy":
        cali = CalibrationMethod.Entropy
    else:
        cali = CalibrationMethod.Percentile
        os.environ["ORT_QUANTIZATION_CALIBRATION_PERCENTILE"] = "99.9"

    nodes_to_exclude = build_exclude_list(args.onnx_in, cfg.get("nodes_to_exclude_substrings", []))
    print(f"[INFO] Excluding {len(nodes_to_exclude)} nodes from quantization.")
    if nodes_to_exclude:
        print("\n".join(nodes_to_exclude[:50]))

    quantize_static(
        model_input=args.onnx_in,
        model_output=args.onnx_out,
        calibration_data_reader=dr,
        per_channel=bool(cfg.get("per_channel", True)),
        activation_type=act_dt,
        weight_type=wt_dt,
        optimize_model=True,
        calibrate_method=cali,
        nodes_to_exclude=nodes_to_exclude
    )
    print(f"[OK] INT8 model written to {args.onnx_out}")
