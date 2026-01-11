"""TensorRT inference runner with dynamic shape support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:  # allow --help without dependencies
    import pycuda.driver as cuda
    import tensorrt as trt
except Exception:  # pragma: no cover
    cuda = None  # type: ignore
    trt = None  # type: ignore


@dataclass
class BindingInfo:
    index: int
    name: str
    is_input: bool
    dtype: np.dtype
    shape: tuple


class TrtRunner:
    """TensorRT runtime wrapper for dynamic batch inference."""

    def __init__(self, engine_path: str, profile_index: int = 0, device_index: int = 0) -> None:
        if trt is None or cuda is None:
            raise ImportError("TensorRT and PyCUDA are required for TrtRunner")
        self.engine_path = engine_path
        self.profile_index = profile_index
        self.device_index = device_index
        self.logger = trt.Logger(trt.Logger.ERROR)
        cuda.init()
        self._device = cuda.Device(device_index)
        self._context = self._device.make_context()
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self._use_binding_api = hasattr(self.engine, "get_binding_name")
        self._bindings: List[BindingInfo] = self._scan_bindings()
        self._host_buffers: Dict[int, np.ndarray] = {}
        self._device_buffers: Dict[int, cuda.DeviceAllocation] = {}
        self._alloc_sizes: Dict[int, int] = {}

        if hasattr(self.context, "set_optimization_profile_async"):
            self.context.set_optimization_profile_async(profile_index, self.stream.handle)

    def close(self) -> None:
        if getattr(self, "_context", None) is not None:
            if getattr(self, "stream", None) is not None:
                try:
                    self.stream.synchronize()
                except Exception:
                    pass
            self._context.pop()
            self._context.detach()
            self._context = None

    def __del__(self) -> None:
        self.close()

    def _scan_bindings(self) -> List[BindingInfo]:
        bindings: List[BindingInfo] = []
        if self._use_binding_api:
            for idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(idx)
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))
                shape = tuple(self.engine.get_binding_shape(idx))
                bindings.append(BindingInfo(idx, name, self.engine.binding_is_input(idx), dtype, shape))
        else:
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = tuple(self.engine.get_tensor_shape(name))
                is_input = mode == trt.TensorIOMode.INPUT
                bindings.append(BindingInfo(idx, name, is_input, dtype, shape))
        return bindings

    @property
    def input_binding(self) -> BindingInfo:
        return next(b for b in self._bindings if b.is_input)

    def input_channels(self) -> int | None:
        shape = self.input_binding.shape
        if len(shape) >= 2 and shape[1] not in (-1, 0):
            return int(shape[1])
        return None

    def dump_bindings(self) -> None:
        print("[TrtRunner] Bindings:")
        for binding in self._bindings:
            if self._use_binding_api:
                engine_shape = tuple(self.engine.get_binding_shape(binding.index))
                ctx_shape = tuple(self.context.get_binding_shape(binding.index))
            else:
                engine_shape = tuple(self.engine.get_tensor_shape(binding.name))
                ctx_shape = tuple(self.context.get_tensor_shape(binding.name))
            print(
                f"  idx={binding.index} name={binding.name} input={binding.is_input} "
                f"dtype={binding.dtype} engine_shape={engine_shape} ctx_shape={ctx_shape}"
            )

    def _ensure_buffers(self, index: int, shape: tuple, dtype: np.dtype) -> None:
        size = int(np.prod(shape))
        needed_bytes = size * np.dtype(dtype).itemsize
        if self._alloc_sizes.get(index, 0) >= needed_bytes:
            return
        host = cuda.pagelocked_empty(size, dtype)
        device = cuda.mem_alloc(needed_bytes)
        self._host_buffers[index] = host
        self._device_buffers[index] = device
        self._alloc_sizes[index] = needed_bytes

    def infer(self, x: np.ndarray, verbose: bool = False) -> Dict[str, np.ndarray]:
        if x.dtype != np.float32:
            raise AssertionError("input dtype must be np.float32")
        if not x.flags["C_CONTIGUOUS"]:
            raise AssertionError("input must be C_CONTIGUOUS")

        input_binding = self.input_binding
        if self._use_binding_api:
            self.context.set_binding_shape(input_binding.index, x.shape)
        else:
            self.context.set_input_shape(input_binding.name, x.shape)

        if hasattr(self.context, "all_binding_shapes_specified"):
            assert self.context.all_binding_shapes_specified

        if verbose:
            print(f"[TrtRunner] input shape: {x.shape}")

        bindings_ptrs = [0] * len(self._bindings)
        outputs: Dict[str, np.ndarray] = {}

        self._ensure_buffers(input_binding.index, x.shape, input_binding.dtype)
        host_in = self._host_buffers[input_binding.index]
        host_in.reshape(x.shape)
        np.copyto(host_in.reshape(x.shape), x)
        cuda.memcpy_htod_async(self._device_buffers[input_binding.index], host_in, self.stream)
        bindings_ptrs[input_binding.index] = int(self._device_buffers[input_binding.index])

        for binding in self._bindings:
            if binding.is_input:
                continue
            if self._use_binding_api:
                out_shape = tuple(self.context.get_binding_shape(binding.index))
            else:
                out_shape = tuple(self.context.get_tensor_shape(binding.name))
            if verbose:
                transpose_hint = "yes" if len(out_shape) == 3 and out_shape[1] < out_shape[2] else "no"
                print(f"[TrtRunner] output {binding.name} shape={out_shape} transpose_hint={transpose_hint}")
            self._ensure_buffers(binding.index, out_shape, binding.dtype)
            bindings_ptrs[binding.index] = int(self._device_buffers[binding.index])

        if hasattr(self.context, "execute_async_v3") and not self._use_binding_api:
            for binding in self._bindings:
                ptr = self._device_buffers[binding.index]
                self.context.set_tensor_address(binding.name, int(ptr))
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(bindings_ptrs, self.stream.handle)

        for binding in self._bindings:
            if binding.is_input:
                continue
            if self._use_binding_api:
                out_shape = tuple(self.context.get_binding_shape(binding.index))
            else:
                out_shape = tuple(self.context.get_tensor_shape(binding.name))
            host_out = self._host_buffers[binding.index]
            cuda.memcpy_dtoh_async(host_out, self._device_buffers[binding.index], self.stream)
            outputs[binding.name] = host_out.reshape(out_shape).copy()

        self.stream.synchronize()
        return outputs
