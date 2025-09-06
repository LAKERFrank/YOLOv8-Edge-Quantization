import argparse
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_static,
)


def letterbox(im, new_size=640, color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding."""
    h, w = im.shape[:2]
    r = new_size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas = np.full((new_size, new_size, 3), color, dtype=im.dtype)
    canvas[top : top + nh, left : left + nw] = im
    return canvas


class YOLOCalibrationDataReader(CalibrationDataReader):
    """DataReader for feeding calibration images to quantize_static."""

    def __init__(self, calib_dir, input_name="images", size=640):
        self.image_paths = sorted(glob.glob(os.path.join(calib_dir, "*")))
        self.input_name = input_name
        self.size = size
        self.i = 0

    def get_next(self):
        if self.i >= len(self.image_paths):
            return None
        img = cv2.imread(self.image_paths[self.i])
        self.i += 1
        img = letterbox(img, self.size)
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return {self.input_name: np.expand_dims(img, 0)}


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 to ONNX and perform static INT8 quantization"
    )
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 .pt weights")
    parser.add_argument(
        "--calib-dir", required=True, help="Directory containing calibration images"
    )
    parser.add_argument(
        "--onnx-out",
        default="model-fp32.onnx",
        help="Path to save the exported FP32 ONNX model",
    )
    parser.add_argument(
        "--quant-out",
        default="model-int8.onnx",
        help="Path to save the quantized INT8 ONNX model",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference size for export and calibration"
    )
    args = parser.parse_args()

    # Export the model to ONNX
    model = YOLO(args.weights)
    onnx_tmp = model.export(
        format="onnx",
        opset=12,
        simplify=True,
        dynamic=False,
        imgsz=args.imgsz,
    )
    # Rename to the desired output path
    os.replace(onnx_tmp, args.onnx_out)

    # Quantize with static post-training quantization
    dr = YOLOCalibrationDataReader(args.calib_dir, size=args.imgsz)
    quantize_static(
        model_input=args.onnx_out,
        model_output=args.quant_out,
        calibration_data_reader=dr,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )
    print(f"[OK] Quantized model saved to {args.quant_out}")


if __name__ == "__main__":
    main()
