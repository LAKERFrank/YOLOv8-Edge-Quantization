import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def preprocess(frames: list[np.ndarray]) -> np.ndarray:
    """
    將 10 張 640x640 灰階圖像 → (1,10,640,640)
    """
    assert len(frames) == 10, "需提供10張灰階圖片"
    processed = [cv2.resize(f, (640, 640)) for f in frames]
    stacked = np.stack(processed, axis=0)  # (10, 640, 640)
    stacked = stacked.astype(np.float32) / 255.0
    stacked = np.expand_dims(stacked, axis=0)  # (1, 10, 640, 640)
    return stacked

def infer(engine, inputs, outputs, bindings, stream, input_tensor: np.ndarray):
    np.copyto(inputs[0]['host'], input_tensor.ravel())

    with engine.create_execution_context() as context:
        context.set_binding_shape(0, input_tensor.shape)

        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        return outputs[0]['host']

# === 主程式入口 ===
engine_path = "your_model.engine"
engine = load_engine(engine_path)
inputs, outputs, bindings, stream = allocate_buffers(engine)

# 讀取 10 張灰階影像（640x640）
frames = [cv2.imread(f"frame_{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(10)]
input_tensor = preprocess(frames)  # shape: (1,10,640,640)

output = infer(engine, inputs, outputs, bindings, stream, input_tensor)
print("Inference result:", output[:10])
