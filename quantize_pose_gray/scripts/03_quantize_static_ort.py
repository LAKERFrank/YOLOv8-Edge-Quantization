import importlib
import inspect
import os
import yaml
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

CalibDataReader1Ch = importlib.import_module("02_gray_calib_reader").CalibDataReader1Ch

FP32_ONNX = "yolov8n-pose-gray.fp32.onnx"
INT8_ONNX = "yolov8n-pose-gray.int8.qdq.onnx"
CONFIG = os.path.join(os.path.dirname(__file__), "..", "quant_config.yaml")

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 1) Determine input tensor name
sess = ort.InferenceSession(FP32_ONNX, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
print("input_name:", input_name)

# 2) Calibration reader
calib_cfg = cfg.get("calib", {})
reader = CalibDataReader1Ch(calib_cfg.get("dir", "data/calib_list.txt"),
                             input_name=input_name,
                             size=calib_cfg.get("size", 640),
                             limit=calib_cfg.get("limit"))

# Warn if calibration set is small
calib_count = len(reader.paths)
print(f"calibration samples: {calib_count}")
if calib_count < 50:
    print("WARNING: calibration set has fewer than 50 images; quantization accuracy may suffer")

qcfg = cfg.get("quant", {})
extra_opts = {
    "ActivationSymmetric": qcfg.get("activation", {}).get("symmetric", False),
    "WeightSymmetric": qcfg.get("weight", {}).get("symmetric", False),
}

print("quantization parameters:")
print(f"  activation: dtype={qcfg.get('activation', {}).get('dtype', 'qint8')}, per_channel={qcfg.get('activation', {}).get('per_channel', False)}, symmetric={extra_opts['ActivationSymmetric']}")
print(f"  weight: dtype={qcfg.get('weight', {}).get('dtype', 'qint8')}, per_channel={qcfg.get('weight', {}).get('per_channel', True)}, symmetric={extra_opts['WeightSymmetric']}")
print(f"  ops: {qcfg.get('ops', ['Conv', 'MatMul'])}")

qs_args = dict(
    model_input=FP32_ONNX,
    model_output=INT8_ONNX,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    per_channel=qcfg.get("weight", {}).get("per_channel", True),
    activation_type=QuantType.QInt8 if qcfg.get("activation", {}).get("dtype", "qint8") == "qint8" else QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=qcfg.get("ops", ["Conv", "MatMul"]),
    extra_options=extra_opts,
)

# nodes to exclude based on substrings and ablation toggles
excludes = set(qcfg.get("nodes_to_exclude_substrings", []))
abl = cfg.get("ablation", {})
if abl.get("dequantize_kpt_head", False):
    excludes.update(["kpt"])
if abl.get("dequantize_bbox_dfl_head", False):
    excludes.update(["bbox", "dfl"])
excludes.update(abl.get("extra_excludes", []))

model = onnx.load(FP32_ONNX)
node_names = []
for n in model.graph.node:
    if any(s in n.name for s in excludes):
        node_names.append(n.name)
if node_names:
    qs_args["nodes_to_exclude"] = node_names

# Some onnxruntime versions support 'optimize_model'
if "optimize_model" in inspect.signature(quantize_static).parameters:
    qs_args["optimize_model"] = True

# 3) Run quantization
quantize_static(**qs_args)
print("Done ->", INT8_ONNX)
