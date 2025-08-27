import argparse
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    """Export a YOLOv8 model to ONNX and perform dynamic INT8 quantization.

    This script follows the steps described in the Medium article by
    `sulavstha007` on quantizing YOLOv8 models.  It first exports a PyTorch
    weight file to an ONNX model and then applies weight-only dynamic
    quantization using ONNX Runtime.
    """
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX and quantize to INT8")
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 .pt weights")
    parser.add_argument("--onnx-out", default="model-fp32.onnx", help="Path to save the exported FP32 ONNX model")
    parser.add_argument("--quant-out", default="model-int8.onnx", help="Path to save the quantized INT8 ONNX model")
    args = parser.parse_args()

    # Export the model to ONNX using Ultralytics
    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        opset=12,
        simplify=True,
        dynamic=False,
        imgsz=640,
        path=args.onnx_out,
    )

    # Quantize the exported model to INT8 with weight-only dynamic quantization
    quantize_dynamic(
        model_input=onnx_path,
        model_output=args.quant_out,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )
    print(f"[OK] Quantized model saved to {args.quant_out}")


if __name__ == "__main__":
    main()
