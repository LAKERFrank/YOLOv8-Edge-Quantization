import argparse, yaml
from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--onnx-in", required=True)
    ap.add_argument("--onnx-out", required=True)
    args = ap.parse_args()
    _ = yaml.safe_load(open(args.cfg))  # reserved for input_name if needed
    quantize_dynamic(
        model_input=args.onnx_in,
        model_output=args.onnx_out,
        weight_type=QuantType.QInt8,
        optimize_model=True
    )
    print(f"[OK] Dynamic INT8 (weights-only) -> {args.onnx_out}")
