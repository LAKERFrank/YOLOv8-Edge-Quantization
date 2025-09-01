import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quantize_static,
)
import importlib
import inspect

GrayCalibReader = importlib.import_module("02_gray_calib_reader").GrayCalibReader

FP32_ONNX = "yolov8n-pose-gray.fp32.onnx"
INT8_ONNX = "yolov8n-pose-gray.int8.qdq.onnx"
CALIB_LIST = "data/calib_list.txt"
IMG_SIZE = 640

# 1) Determine input tensor name
sess = ort.InferenceSession(FP32_ONNX, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
print("input_name:", input_name)

# 2) Create grayscale calibration reader
reader = GrayCalibReader(CALIB_LIST, input_name=input_name, size=IMG_SIZE)

qs_args = dict(
    model_input=FP32_ONNX,
    model_output=INT8_ONNX,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["Conv", "MatMul"],
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    },
)

# Some onnxruntime versions support 'optimize_model'; enable if available.
if "optimize_model" in inspect.signature(quantize_static).parameters:
    qs_args["optimize_model"] = True

# 3) Run QDQ static quantization
quantize_static(**qs_args)

print("Done ->", INT8_ONNX)
