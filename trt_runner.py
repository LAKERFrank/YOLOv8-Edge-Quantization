import dataclasses
from typing import Dict, List, Tuple

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


@dataclasses.dataclass
class BindingInfo:
    index: int
    name: str
    is_input: bool
    dtype: np.dtype


class TrtRunner:
    def __init__(self, engine_path: str, profile_index: int = 0) -> None:
        self.engine_path = engine_path
        self.profile_index = profile_index
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        cuda.init()
        self.stream = cuda.Stream()
        if self.engine.num_optimization_profiles > 0:
            if hasattr(self.context, "set_optimization_profile_async"):
                self.context.set_optimization_profile_async(profile_index, self.stream.handle)
            else:
                try:
                    self.context.active_optimization_profile = profile_index
                except AttributeError:
                    pass
        self.bindings: List[BindingInfo] = []
        for idx, name, is_input, dtype in self._iter_bindings():
            self.bindings.append(BindingInfo(idx, name, is_input, dtype))
        self._host_buffers: Dict[int, np.ndarray] = {}
        self._device_buffers: Dict[int, cuda.DeviceAllocation] = {}
        self._buffer_sizes: Dict[int, int] = {}

    def _iter_bindings(self):
        if hasattr(self.engine, "num_bindings"):
            for idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(idx)
                is_input = self.engine.binding_is_input(idx)
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))
                yield idx, name, is_input, dtype
        else:
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                is_input = mode == trt.TensorIOMode.INPUT
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                yield idx, name, is_input, dtype

    def _get_binding_shape(self, binding_index: int, binding_name: str) -> Tuple[int, ...]:
        if hasattr(self.context, "get_binding_shape"):
            return tuple(self.context.get_binding_shape(binding_index))
        return tuple(self.context.get_tensor_shape(binding_name))

    def _set_binding_shape(self, binding_index: int, binding_name: str, shape: Tuple[int, ...]) -> None:
        if hasattr(self.context, "set_binding_shape"):
            self.context.set_binding_shape(binding_index, shape)
        else:
            self.context.set_input_shape(binding_name, shape)

    def _get_num_bindings(self) -> int:
        if hasattr(self.engine, "num_bindings"):
            return self.engine.num_bindings
        return self.engine.num_io_tensors

    def get_input_channel_count(self) -> int | None:
        input_binding = next(b for b in self.bindings if b.is_input)
        if hasattr(self.engine, "get_binding_shape"):
            shape = tuple(self.engine.get_binding_shape(input_binding.index))
        else:
            shape = tuple(self.engine.get_tensor_shape(input_binding.name))
        if len(shape) >= 2:
            return shape[1]
        return None

    def dump_bindings(self) -> None:
        print("[TrtRunner] Bindings (engine shapes):")
        for binding in self.bindings:
            if hasattr(self.engine, "get_binding_shape"):
                engine_shape = self.engine.get_binding_shape(binding.index)
            else:
                engine_shape = self.engine.get_tensor_shape(binding.name)
            print(
                f"  - {binding.name}: is_input={binding.is_input}, "
                f"dtype={binding.dtype}, shape={engine_shape}"
            )
        if self.context is not None:
            print("[TrtRunner] Bindings (context shapes):")
            for binding in self.bindings:
                ctx_shape = self._get_binding_shape(binding.index, binding.name)
                print(
                    f"  - {binding.name}: is_input={binding.is_input}, "
                    f"dtype={binding.dtype}, shape={ctx_shape}"
                )

    def _allocate_if_needed(self, binding: BindingInfo, shape: Tuple[int, ...]) -> None:
        size = int(np.prod(shape))
        current_size = self._buffer_sizes.get(binding.index, 0)
        if size <= current_size:
            return
        host_buf = cuda.pagelocked_empty(size, binding.dtype)
        device_buf = cuda.mem_alloc(host_buf.nbytes)
        self._host_buffers[binding.index] = host_buf
        self._device_buffers[binding.index] = device_buf
        self._buffer_sizes[binding.index] = size

    def infer(self, x: np.ndarray, verbose: bool = False) -> Dict[str, np.ndarray]:
        assert x.dtype == np.float32, "input dtype must be float32"
        assert x.flags["C_CONTIGUOUS"], "input must be C_CONTIGUOUS"
        input_binding = next(b for b in self.bindings if b.is_input)
        self._set_binding_shape(input_binding.index, input_binding.name, x.shape)
        assert self.context.all_binding_shapes_specified
        if verbose:
            print(f"[TrtRunner] input shape: {x.shape}")
        outputs: Dict[str, np.ndarray] = {}
        bindings_ptrs: List[int] = [0] * self._get_num_bindings()
        for binding in self.bindings:
            binding_shape = self._get_binding_shape(binding.index, binding.name)
            self._allocate_if_needed(binding, binding_shape)
            bindings_ptrs[binding.index] = int(self._device_buffers[binding.index])
            if binding.is_input:
                np.copyto(self._host_buffers[binding.index][: x.size], x.ravel())
                cuda.memcpy_htod_async(
                    self._device_buffers[binding.index],
                    self._host_buffers[binding.index][: x.size],
                    self.stream,
                )
            else:
                if verbose:
                    print(f"[TrtRunner] output {binding.name} shape: {binding_shape}")
                    if len(binding_shape) == 3 and binding_shape[1] < binding_shape[2]:
                        print(
                            f"[TrtRunner] output {binding.name}: "
                            "transpose may be required based on shape"
                        )
        self.context.execute_async_v2(bindings=bindings_ptrs, stream_handle=self.stream.handle)
        for binding in self.bindings:
            if binding.is_input:
                continue
            binding_shape = self._get_binding_shape(binding.index, binding.name)
            size = int(np.prod(binding_shape))
            cuda.memcpy_dtoh_async(
                self._host_buffers[binding.index][:size],
                self._device_buffers[binding.index],
                self.stream,
            )
        self.stream.synchronize()
        for binding in self.bindings:
            if binding.is_input:
                continue
            binding_shape = self._get_binding_shape(binding.index, binding.name)
            size = int(np.prod(binding_shape))
            host_buf = self._host_buffers[binding.index][:size]
            outputs[binding.name] = host_buf.reshape(binding_shape).copy()
        return outputs
