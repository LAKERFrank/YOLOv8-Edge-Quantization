#!/usr/bin/env python3
"""Compare TensorRT engine outputs between a reference and a test engine."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover - depends on runtime
    raise SystemExit(
        "TensorRT Python package is required to run compare_model_outputs.py"
    ) from exc

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  # side effect: initialise CUDA context
except ImportError as exc:  # pragma: no cover - depends on runtime
    raise SystemExit("pycuda is required to run compare_model_outputs.py") from exc


@dataclass
class BindingInfo:
    name: str
    index: int
    dtype: np.dtype
    is_input: bool


class EngineIOHelper:
    """Compatibility layer between legacy binding APIs and TensorRT 10 IO tensors."""

    def __init__(self, engine: "trt.ICudaEngine") -> None:
        self.engine = engine
        self.tensor_io_mode = getattr(trt, "TensorIOMode", None)
        self.execution_status = getattr(trt, "ExecutionStatus", None)
        self.using_bindings = hasattr(engine, "num_bindings") and hasattr(
            engine, "get_binding_name"
        )
        self.using_io_tensors = hasattr(engine, "num_io_tensors") and hasattr(
            engine, "get_tensor_name"
        )
        if not (self.using_bindings or self.using_io_tensors):
            raise RuntimeError(
                "TensorRT engine exposes neither legacy binding APIs nor IO tensor APIs"
            )
        self._all_names = self._collect_names()
        self._name_to_index = {name: idx for idx, name in enumerate(self._all_names)}

    def _collect_names(self) -> List[str]:
        names: List[str] = []
        if self.using_bindings:
            count = self._get_num_bindings_raw()
            for idx in range(count):
                names.append(self.engine.get_binding_name(idx))
        else:
            count = self._get_num_io_tensors_raw()
            get_tensor_name = getattr(self.engine, "get_tensor_name")
            for idx in range(count):
                try:
                    name = get_tensor_name(idx)
                except TypeError:
                    # Some TensorRT builds expect the tensor name instead of index; ignore.
                    name = None
                if not name:
                    continue
                names.append(name)
        return names

    def _get_num_bindings_raw(self) -> int:
        attr = getattr(self.engine, "num_bindings", None)
        if attr is None:
            return 0
        return int(attr() if callable(attr) else attr)

    def _get_num_io_tensors_raw(self) -> int:
        attr = getattr(self.engine, "num_io_tensors", None)
        if attr is None:
            return 0
        return int(attr() if callable(attr) else attr)

    @staticmethod
    def _mode_matches(mode: object, target: str) -> bool:
        if mode is None:
            return False
        try:
            # Direct enum comparison when TensorIOMode is available.
            enum_value = getattr(trt.TensorIOMode, target)
        except AttributeError:
            enum_value = None
        if enum_value is not None:
            try:
                return mode == enum_value
            except Exception:  # pragma: no cover - defensive against mismatched enums
                pass
        return target.lower() in str(mode).lower()

    def _dims_to_tuple(self, dims: Any) -> Tuple[int, ...]:
        if dims is None:
            return ()
        if isinstance(dims, tuple):
            return tuple(int(d) for d in dims)
        if isinstance(dims, list):
            return tuple(int(d) for d in dims)
        try:
            return tuple(int(dims[i]) for i in range(len(dims)))
        except Exception:  # pragma: no cover - fallback for unexpected containers
            return tuple(int(v) for v in dims)

    @property
    def io_tensor_names(self) -> List[str]:
        names: List[str] = []
        for name in self._all_names:
            if self.is_shape_tensor(name):
                continue
            if self.using_bindings:
                names.append(name)
            else:
                if self.is_input(name) or self.is_output(name):
                    names.append(name)
        return names

    @property
    def input_names(self) -> List[str]:
        return [name for name in self.io_tensor_names if self.is_input(name)]

    @property
    def output_names(self) -> List[str]:
        return [name for name in self.io_tensor_names if self.is_output(name)]

    def num_bindings(self) -> int:
        if self.using_bindings:
            return self._get_num_bindings_raw()
        return len(self.io_tensor_names)

    def get_binding_index(self, name: str) -> int:
        if self.using_bindings and hasattr(self.engine, "get_binding_index"):
            idx = self.engine.get_binding_index(name)
            if idx != -1:
                return idx
        if hasattr(self.engine, "get_tensor_index"):
            try:
                idx = self.engine.get_tensor_index(name)
                if idx != -1:
                    return idx
            except TypeError:
                pass
        return self._name_to_index.get(name, -1)

    def is_input(self, name: str) -> bool:
        if self.using_bindings and hasattr(self.engine, "binding_is_input"):
            idx = self.get_binding_index(name)
            if idx == -1:
                return False
            return bool(self.engine.binding_is_input(idx))
        mode_fn = getattr(self.engine, "get_tensor_mode", None)
        if mode_fn is None:
            return False
        mode = mode_fn(name)
        return self._mode_matches(mode, "INPUT") or self._mode_matches(mode, "BIDIRECTIONAL")

    def is_output(self, name: str) -> bool:
        if self.using_bindings:
            return not self.is_input(name)
        mode_fn = getattr(self.engine, "get_tensor_mode", None)
        if mode_fn is None:
            return False
        mode = mode_fn(name)
        return self._mode_matches(mode, "OUTPUT") or self._mode_matches(mode, "BIDIRECTIONAL")

    def is_shape_tensor(self, name: str) -> bool:
        mode_fn = getattr(self.engine, "get_tensor_mode", None)
        if mode_fn is None:
            return False
        mode = mode_fn(name)
        return self._mode_matches(mode, "SHAPE")

    def get_dtype(self, name: str) -> np.dtype:
        if self.using_bindings and hasattr(self.engine, "get_binding_index"):
            idx = self.engine.get_binding_index(name)
            if idx != -1:
                return np.dtype(trt.nptype(self.engine.get_binding_dtype(idx)))
        if hasattr(self.engine, "get_tensor_dtype"):
            return np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
        raise RuntimeError(f"Unable to determine dtype for binding '{name}'")

    def get_engine_shape(self, name: str) -> Tuple[int, ...]:
        if self.using_bindings and hasattr(self.engine, "get_binding_index"):
            idx = self.engine.get_binding_index(name)
            if idx != -1:
                return self._dims_to_tuple(self.engine.get_binding_shape(idx))
        if hasattr(self.engine, "get_tensor_shape"):
            return self._dims_to_tuple(self.engine.get_tensor_shape(name))
        return ()

    def get_context_shape(self, context: "trt.IExecutionContext", name: str) -> Tuple[int, ...]:
        if self.using_bindings and hasattr(context, "get_binding_shape"):
            idx = self.get_binding_index(name)
            if idx != -1:
                return self._dims_to_tuple(context.get_binding_shape(idx))
        if hasattr(context, "get_tensor_shape"):
            return self._dims_to_tuple(context.get_tensor_shape(name))
        return ()

    def set_context_shape(
        self, context: "trt.IExecutionContext", name: str, shape: Tuple[int, ...]
    ) -> None:
        if self.using_bindings and hasattr(context, "set_binding_shape"):
            idx = self.get_binding_index(name)
            if idx == -1:
                raise RuntimeError(f"Unknown binding '{name}'")
            context.set_binding_shape(idx, tuple(int(d) for d in shape))
            return
        setter = getattr(context, "set_input_shape", None)
        if setter is None:
            setter = getattr(context, "set_tensor_shape", None)
        if setter is None:
            raise AttributeError(
                "Execution context does not provide set_input_shape/set_tensor_shape APIs"
            )
        try:
            setter(name, tuple(int(d) for d in shape))
        except TypeError:
            idx = self.get_binding_index(name)
            setter(idx, tuple(int(d) for d in shape))

    def all_input_shapes_specified(self, context: "trt.IExecutionContext") -> bool:
        attr = None
        if self.using_bindings and hasattr(context, "all_binding_shapes_specified"):
            attr = context.all_binding_shapes_specified
        elif hasattr(context, "all_input_shapes_specified"):
            attr = context.all_input_shapes_specified
        if attr is None:
            return True
        if callable(attr):
            attr = attr()
        return bool(attr)

    def get_profile_opt_shape(self, name: str, profile_index: int) -> Optional[Tuple[int, ...]]:
        if hasattr(self.engine, "get_profile_shape"):
            idx = self.get_binding_index(name)
            if idx != -1:
                try:
                    shapes = self.engine.get_profile_shape(profile_index, idx)
                except TypeError:
                    shapes = self.engine.get_profile_shape(idx)
                if shapes:
                    try:
                        return self._dims_to_tuple(shapes[1])
                    except Exception:
                        pass
        if hasattr(self.engine, "get_tensor_profile_shape"):
            try:
                shapes = self.engine.get_tensor_profile_shape(name, profile_index)
            except TypeError:
                shapes = self.engine.get_tensor_profile_shape(name)
            if shapes:
                try:
                    return self._dims_to_tuple(shapes[1])
                except Exception:
                    pass
        return None

    def set_tensor_address(
        self, context: "trt.IExecutionContext", name: str, device_ptr: int
    ) -> None:
        setter = getattr(context, "set_tensor_address", None)
        if setter is None:
            raise AttributeError(
                "Execution context does not provide set_tensor_address required for IO tensors"
            )
        try:
            setter(name, device_ptr)
        except TypeError:
            idx = self.get_binding_index(name)
            setter(idx, device_ptr)

    def set_optimization_profile(
        self,
        context: "trt.IExecutionContext",
        profile_index: int,
        stream_handle: Optional[int] = None,
    ) -> None:
        num_profiles_attr = getattr(self.engine, "num_optimization_profiles", 0)
        if callable(num_profiles_attr):
            try:
                num_profiles = int(num_profiles_attr())
            except TypeError:
                num_profiles = int(num_profiles_attr)
        else:
            num_profiles = int(num_profiles_attr)
        if num_profiles <= 0:
            return
        if not 0 <= profile_index < num_profiles:
            raise ValueError(
                f"Profile index {profile_index} out of range for engine with {num_profiles} profiles"
            )

        set_async = getattr(context, "set_optimization_profile_async", None)
        if set_async is not None:
            temp_stream = None
            handle = stream_handle
            if handle is None:
                temp_stream = cuda.Stream()
                handle = temp_stream.handle
            try:
                set_async(profile_index, handle)
                if temp_stream is not None:
                    temp_stream.synchronize()
            finally:
                if temp_stream is not None:
                    del temp_stream
            return

        set_sync = getattr(context, "set_optimization_profile", None)
        if set_sync is not None:
            set_sync(profile_index)
            return

        current = getattr(context, "active_optimization_profile", None)
        if current is not None:
            try:
                if callable(current):
                    current = current()
            except TypeError:
                pass
            try:
                if int(current) == int(profile_index):
                    return
            except (TypeError, ValueError):
                pass

        if profile_index == 0:
            # Many runtimes default to profile 0 and expose only a read-only attribute.
            return

        raise RuntimeError("Unable to select optimization profile on execution context")

    @property
    def requires_tensor_io(self) -> bool:
        return not self.using_bindings

def load_engine(engine_path: Path) -> trt.ICudaEngine:
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine from {engine_path}")
    return engine


def parse_shape_string(shape_str: str) -> Tuple[str, Tuple[int, ...]]:
    if ":" not in shape_str:
        raise ValueError(f"Invalid shape specification '{shape_str}'. Expected format name:dim,dim,..")
    name, dims_part = shape_str.split(":", 1)
    dims_part = dims_part.strip().lower().replace("x", ",")
    dims = [d for d in dims_part.replace("(", "").replace(")", "").split(",") if d]
    if not dims:
        raise ValueError(f"Invalid shape specification '{shape_str}'")
    try:
        dims_tuple = tuple(int(d.strip()) for d in dims)
    except ValueError as exc:
        raise ValueError(f"Invalid dimension in '{shape_str}'") from exc
    return name.strip(), dims_tuple


def build_binding_info(engine: trt.ICudaEngine) -> Dict[str, BindingInfo]:
    helper = EngineIOHelper(engine)
    info: Dict[str, BindingInfo] = {}
    for name in helper.io_tensor_names:
        info[name] = BindingInfo(
            name=name,
            index=helper.get_binding_index(name),
            dtype=helper.get_dtype(name),
            is_input=helper.is_input(name),
        )
    return info


def get_binding_names(engine: trt.ICudaEngine, *, inputs: bool) -> List[str]:
    helper = EngineIOHelper(engine)
    return helper.input_names if inputs else helper.output_names


def ensure_engines_compatible(
    ref_info: Mapping[str, BindingInfo], test_info: Mapping[str, BindingInfo]
) -> None:
    ref_inputs = {name for name, b in ref_info.items() if b.is_input}
    test_inputs = {name for name, b in test_info.items() if b.is_input}
    if ref_inputs != test_inputs:
        missing = ref_inputs.symmetric_difference(test_inputs)
        raise ValueError(f"Input bindings mismatch between engines: {sorted(missing)}")

    ref_outputs = {name for name, b in ref_info.items() if not b.is_input}
    test_outputs = {name for name, b in test_info.items() if not b.is_input}
    if ref_outputs != test_outputs:
        missing = ref_outputs.symmetric_difference(test_outputs)
        raise ValueError(f"Output bindings mismatch between engines: {sorted(missing)}")

    for name, ref_binding in ref_info.items():
        test_binding = test_info.get(name)
        if test_binding is None:
            continue
        if ref_binding.dtype != test_binding.dtype:
            raise ValueError(
                f"Binding '{name}' dtype mismatch: {ref_binding.dtype} vs {test_binding.dtype}"
            )


def resolve_input_shapes(
    engine: trt.ICudaEngine,
    provided: Mapping[str, Tuple[int, ...]],
    batch_size: int,
    profile_index: int,
) -> Dict[str, Tuple[int, ...]]:
    helper = EngineIOHelper(engine)
    context = engine.create_execution_context()
    try:
        helper.set_optimization_profile(context, profile_index)
    except RuntimeError:
        current = getattr(context, "active_optimization_profile", None)
        if callable(current):  # pragma: no cover - defensive
            try:
                current = current()
            except TypeError:
                current = None
        matches = False
        if current is not None:
            try:
                matches = int(current) == int(profile_index)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                matches = False
        if not matches and profile_index != 0:
            raise
    shapes: Dict[str, Tuple[int, ...]] = {}
    implicit = engine.has_implicit_batch_dimension
    for name in helper.input_names:
        if name in provided:
            shapes[name] = tuple(provided[name])
            continue

        if implicit:
            base_dims = helper.get_engine_shape(name)
            dims: Tuple[int, ...] = (batch_size, *tuple(int(d) for d in base_dims))
        else:
            ctx_dims = helper.get_context_shape(context, name)
            if not ctx_dims:
                ctx_dims = helper.get_engine_shape(name)
            dims = tuple(int(d) for d in ctx_dims)

        if any(dim == -1 for dim in dims):
            opt_shape = helper.get_profile_opt_shape(name, profile_index)
            if opt_shape is not None and -1 not in opt_shape:
                if implicit:
                    dims = (batch_size, *tuple(int(d) for d in opt_shape))
                else:
                    dims = tuple(int(d) for d in opt_shape)
            else:
                raise ValueError(
                    f"Input '{name}' has dynamic dimensions. Provide --input-shape {name}:dim,..."
                )
        shapes[name] = tuple(int(d) for d in dims)
    del context
    return shapes


def generate_random_inputs(
    input_info: Mapping[str, BindingInfo],
    shapes: Mapping[str, Tuple[int, ...]],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    inputs: Dict[str, np.ndarray] = {}
    for name, binding in input_info.items():
        if not binding.is_input:
            continue
        shape = shapes[name]
        dtype = binding.dtype
        if np.issubdtype(dtype, np.floating):
            data = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            low = max(info.min, -8)
            high = min(info.max, 8)
            if low >= high:
                low = info.min
                high = info.max
            data = rng.integers(low=low, high=high + 1, size=shape, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            data = rng.integers(0, 2, size=shape, dtype=np.int32).astype(np.bool_)
        else:
            # Fallback for other dtypes: generate float32 and cast
            data = rng.standard_normal(shape).astype(np.float32).astype(dtype)
        inputs[name] = data
    return inputs


def load_torch_module(module_path: Path, device: str):
    import torch

    resolved = module_path.resolve()

    try:
        module = torch.jit.load(str(resolved), map_location=device)
    except Exception as exc:
        try:
            loaded = torch.load(str(resolved), map_location=device)
        except Exception as load_exc:  # pragma: no cover - depends on runtime files
            raise RuntimeError(
                f"Failed to load TorchScript/PyTorch module from {resolved}"
            ) from load_exc
        if not isinstance(loaded, torch.nn.Module):
            raise TypeError(
                "Loaded PyTorch object is not an nn.Module. Provide a TorchScript file or serialized module."
            ) from exc
        module = loaded
    module.eval()
    module.to(device)
    return module


class TRTEngineRunner:
    def __init__(
        self,
        engine: trt.ICudaEngine,
        batch_size: int,
        profile_index: int,
    ) -> None:
        self.engine = engine
        self.batch_size = batch_size
        self.implicit_batch = engine.has_implicit_batch_dimension
        self.helper = EngineIOHelper(engine)
        self.stream = cuda.Stream()
        self.context = engine.create_execution_context()
        try:
            self.helper.set_optimization_profile(
                self.context, profile_index, self.stream.handle
            )
        except RuntimeError:
            current = getattr(self.context, "active_optimization_profile", None)
            if callable(current):  # pragma: no cover - defensive
                try:
                    current = current()
                except TypeError:
                    current = None
            matches = False
            if current is not None:
                try:
                    matches = int(current) == int(profile_index)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    matches = False
            if not matches and profile_index != 0:
                raise
        self.use_tensor_io = self.helper.requires_tensor_io
        if not self.use_tensor_io and not hasattr(self.context, "execute_async_v2"):
            self.use_tensor_io = hasattr(self.context, "set_tensor_address")

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        helper = self.helper
        device_allocations: List[cuda.DeviceAllocation] = []
        host_outputs: Dict[str, np.ndarray] = {}
        output_allocations: Dict[str, cuda.DeviceAllocation] = {}
        prepared_inputs: Dict[str, np.ndarray] = {}
        bindings: Optional[List[int]] = None
        if not self.use_tensor_io:
            bindings = [0] * helper.num_bindings()

        for name in helper.input_names:
            if name not in inputs:
                raise KeyError(f"Missing input '{name}' for inference")
            arr = np.asarray(inputs[name])
            dtype = helper.get_dtype(name)
            if arr.dtype != dtype:
                arr = arr.astype(dtype)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            if self.implicit_batch:
                expected_dims = helper.get_engine_shape(name)
                expected_shape = (self.batch_size, *tuple(int(d) for d in expected_dims))
                if tuple(arr.shape) != expected_shape:
                    raise ValueError(
                        f"Input '{name}' expected shape {expected_shape} for implicit batch engine, got {arr.shape}"
                    )
            prepared_inputs[name] = arr
            if not self.implicit_batch:
                helper.set_context_shape(self.context, name, tuple(int(d) for d in arr.shape))

        if not self.implicit_batch and not helper.all_input_shapes_specified(self.context):
            missing = [
                name
                for name in helper.input_names
                if any(dim == -1 for dim in helper.get_context_shape(self.context, name))
            ]
            raise RuntimeError(f"Binding shapes not fully specified: {missing}")

        # Allocate buffers and copy inputs
        for name in helper.input_names:
            arr = prepared_inputs[name]
            device_mem = cuda.mem_alloc(arr.nbytes)
            device_allocations.append(device_mem)
            cuda.memcpy_htod_async(device_mem, arr, self.stream)
            if self.use_tensor_io:
                helper.set_tensor_address(self.context, name, int(device_mem))
            else:
                assert bindings is not None
                bindings[helper.get_binding_index(name)] = int(device_mem)

        for name in helper.output_names:
            dtype = helper.get_dtype(name)
            if self.implicit_batch:
                dims = helper.get_engine_shape(name)
                out_shape = (self.batch_size, *tuple(int(d) for d in dims))
            else:
                out_shape = helper.get_context_shape(self.context, name)
            if not out_shape or any(dim == -1 for dim in out_shape):
                raise RuntimeError(
                    f"Output binding '{name}' has unresolved shape {out_shape}"
                )
            host_array = np.empty(out_shape, dtype=dtype)
            device_mem = cuda.mem_alloc(host_array.nbytes)
            device_allocations.append(device_mem)
            host_outputs[name] = host_array
            output_allocations[name] = device_mem
            if self.use_tensor_io:
                helper.set_tensor_address(self.context, name, int(device_mem))
            else:
                assert bindings is not None
                bindings[helper.get_binding_index(name)] = int(device_mem)

        # Execute
        if self.use_tensor_io:
            execute_async_v3 = getattr(self.context, "execute_async_v3", None)
            if execute_async_v3 is None:
                raise RuntimeError(
                    "TensorRT runtime does not expose execute_async_v3 required for IO tensors"
                )
            status = execute_async_v3(self.stream.handle)
            status_enum = getattr(trt, "ExecutionStatus", None)
            if status_enum is not None and hasattr(status_enum, "SUCCESS"):
                success = status == status_enum.SUCCESS
            else:
                success = bool(status)
        else:
            assert bindings is not None
            if self.implicit_batch:
                success = self.context.execute_async(
                    batch_size=self.batch_size,
                    bindings=bindings,
                    stream_handle=self.stream.handle,
                )
            else:
                if hasattr(self.context, "execute_async_v2"):
                    success = self.context.execute_async_v2(
                        bindings=bindings, stream_handle=self.stream.handle
                    )
                else:
                    success = self.context.execute_async(
                        batch_size=self.batch_size,
                        bindings=bindings,
                        stream_handle=self.stream.handle,
                    )
        if not success:
            raise RuntimeError("Engine execution failed")

        for name, device_mem in output_allocations.items():
            cuda.memcpy_dtoh_async(host_outputs[name], device_mem, self.stream)
        self.stream.synchronize()
        return host_outputs


class TorchModuleRunner:
    def __init__(
        self,
        module_path: Path,
        input_bindings: Mapping[str, BindingInfo],
        output_names: Sequence[str],
        device: str,
        input_style: str,
    ) -> None:
        import torch

        if input_style not in {"auto", "positional", "named"}:
            raise ValueError("input_style must be one of {'auto', 'positional', 'named'}")
        self.module_path = Path(module_path)
        self.device = torch.device(device)
        self.module = load_torch_module(self.module_path, str(self.device))
        self.input_order = [
            name for name, binding in input_bindings.items() if binding.is_input
        ]
        self.input_dtypes: Dict[str, np.dtype] = {
            name: binding.dtype for name, binding in input_bindings.items() if binding.is_input
        }
        self.output_names = list(output_names)
        self.input_style = input_style
        self.module.eval()
        self.module.to(self.device)

    def _tensor_to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Reference module returned a non-tensor output; only tensors are supported."
            )
        return tensor.detach().to("cpu").numpy()

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        import torch

        torch_inputs: List["torch.Tensor"] = []
        torch_kwargs: Dict[str, "torch.Tensor"] = {}

        for name in self.input_order:
            if name not in inputs:
                raise KeyError(f"Missing input '{name}' for Torch reference module")
            arr = np.asarray(inputs[name])
            expected_dtype = self.input_dtypes[name]
            if arr.dtype != expected_dtype:
                arr = arr.astype(expected_dtype)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            tensor = torch.from_numpy(arr).to(self.device)
            torch_inputs.append(tensor)
            torch_kwargs[name] = tensor

        with torch.no_grad():
            if self.input_style == "named":
                outputs = self.module(**torch_kwargs)
            elif self.input_style == "positional":
                outputs = self.module(*torch_inputs)
            else:
                try:
                    outputs = self.module(**torch_kwargs)
                except TypeError:
                    outputs = self.module(*torch_inputs)

        result: Dict[str, np.ndarray] = {}
        if isinstance(outputs, torch.Tensor):
            if not self.output_names:
                raise ValueError("Torch reference module produced a tensor but no output bindings are defined")
            result[self.output_names[0]] = self._tensor_to_numpy(outputs)
            return result

        if isinstance(outputs, MappingABC):
            for name, tensor in outputs.items():
                if name in self.output_names:
                    result[name] = self._tensor_to_numpy(tensor)
            missing = [name for name in self.output_names if name not in result]
            if missing:
                raise KeyError(
                    "Torch reference module did not return tensors for outputs: "
                    + ", ".join(missing)
                )
            return result

        if isinstance(outputs, SequenceABC):
            if len(outputs) != len(self.output_names):
                raise ValueError(
                    "Torch reference module returned a sequence whose length does not match the engine outputs"
                )
            for name, tensor in zip(self.output_names, outputs):
                result[name] = self._tensor_to_numpy(tensor)
            return result

        raise TypeError(
            "Torch reference module returned unsupported output type. Use tensors, dicts, or sequences of tensors."
        )


def compute_diff(
    reference_outputs: Mapping[str, np.ndarray],
    test_outputs: Mapping[str, np.ndarray],
) -> Dict[str, Mapping[str, float]]:
    diffs: Dict[str, Mapping[str, float]] = {}
    for name, ref_array in reference_outputs.items():
        if name not in test_outputs:
            diffs[name] = {"error": "missing_output"}
            continue
        ref = ref_array.astype(np.float32).ravel()
        test = np.asarray(test_outputs[name]).astype(np.float32).ravel()
        n = min(ref.size, test.size)
        if n == 0:
            diffs[name] = {"error": "empty_output"}
            continue
        ref = ref[:n]
        test = test[:n]
        abs_diff = np.abs(ref - test)
        mae = float(np.mean(abs_diff))
        max_diff = float(np.max(abs_diff))
        diffs[name] = {
            "mae": mae,
            "max": max_diff,
            "mae_per_element": mae,
        }
    return diffs


def aggregate_diffs(diff_records: Sequence[Mapping[str, Mapping[str, float]]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for name in sorted({key for record in diff_records for key in record.keys()}):
        maes: List[float] = []
        maxs: List[float] = []
        errors: List[str] = []
        for record in diff_records:
            entry = record.get(name)
            if not entry:
                continue
            if "error" in entry:
                errors.append(str(entry["error"]))
            else:
                maes.append(float(entry.get("mae", 0.0)))
                maxs.append(float(entry.get("max", 0.0)))
        if maes or errors:
            name_summary: Dict[str, object] = {}
            if maes:
                name_summary["mae_mean"] = float(np.mean(maes))
                name_summary["mae_max"] = float(np.max(maes))
                name_summary["max_abs_mean"] = float(np.mean(maxs))
                name_summary["max_abs_max"] = float(np.max(maxs))
                name_summary["samples"] = len(maes)
            if errors:
                name_summary["errors"] = errors
            summary[name] = name_summary
    return summary


def save_sample_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    sanitized = {name: np.asarray(arr) for name, arr in arrays.items()}
    np.savez(path, **sanitized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare outputs between a TensorRT engine and a reference implementation "
            "(another TensorRT engine or a PyTorch module)."
        )
    )
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "--ref_engine", type=Path, help="Reference (e.g., FP32) TensorRT engine"
    )
    ref_group.add_argument(
        "--ref_torch",
        type=Path,
        help="Reference PyTorch/TorchScript module (.pt/.pth) for comparison",
    )
    parser.add_argument("--test_engine", required=True, type=Path, help="Test (e.g., INT8) engine")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for implicit-batch engines")
    parser.add_argument("--n", type=int, default=10, help="Number of random samples to evaluate")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")
    parser.add_argument(
        "--input-shape",
        action="append",
        default=[],
        help="Override input shape, format name:dim,dim or name:dimxdim",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/compare"),
        help="Directory to store comparison artifacts",
    )
    parser.add_argument(
        "--profile-index",
        type=int,
        default=0,
        help="Optimization profile index to use when executing engines",
    )
    parser.add_argument(
        "--skip-output-save",
        action="store_true",
        help="Skip saving per-sample input/output NPZ files",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default=None,
        help="Device to execute the PyTorch reference module (default: cuda if available)",
    )
    parser.add_argument(
        "--torch-input-style",
        type=str,
        choices=["auto", "positional", "named"],
        default="auto",
        help="How to feed tensors into the PyTorch reference module",
    )
    args = parser.parse_args()

    provided_shapes: Dict[str, Tuple[int, ...]] = {}
    for entry in args.input_shape:
        name, dims = parse_shape_string(entry)
        provided_shapes[name] = dims

    args.outdir.mkdir(parents=True, exist_ok=True)

    test_engine = load_engine(args.test_engine)

    test_bindings = build_binding_info(test_engine)
    test_shapes = resolve_input_shapes(test_engine, provided_shapes, args.batch, args.profile_index)

    if args.ref_engine:
        ref_engine = load_engine(args.ref_engine)
        ref_bindings = build_binding_info(ref_engine)
        ensure_engines_compatible(ref_bindings, test_bindings)
        ref_shapes = resolve_input_shapes(
            ref_engine, provided_shapes, args.batch, args.profile_index
        )
        for name, shape in ref_shapes.items():
            if name not in test_shapes:
                raise ValueError(f"Input '{name}' missing in test engine")
            if tuple(test_shapes[name]) != tuple(shape):
                raise ValueError(
                    f"Input shape mismatch for '{name}': {shape} vs {test_shapes[name]}"
                )
        reference_runner: Any = TRTEngineRunner(ref_engine, args.batch, args.profile_index)
        reference_type = "tensorrt"
        reference_path = Path(args.ref_engine).resolve()
        random_input_bindings = ref_bindings
        torch_device = None
    else:
        ref_shapes = test_shapes
        reference_type = "pytorch"
        reference_path = Path(args.ref_torch).resolve()
        if args.torch_device is None:
            import torch

            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            torch_device = args.torch_device
        reference_runner = TorchModuleRunner(
            module_path=Path(args.ref_torch),
            input_bindings=test_bindings,
            output_names=get_binding_names(test_engine, inputs=False),
            device=torch_device,
            input_style=args.torch_input_style,
        )
        random_input_bindings = test_bindings

    rng = np.random.default_rng(args.seed)

    test_runner = TRTEngineRunner(test_engine, args.batch, args.profile_index)

    diff_records: List[Dict[str, Mapping[str, float]]] = []
    samples_dir = args.outdir / "samples"
    if not args.skip_output_save:
        samples_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        inputs = generate_random_inputs(random_input_bindings, ref_shapes, rng)
        ref_outputs = reference_runner.infer(inputs)
        test_outputs = test_runner.infer(inputs)
        diff = compute_diff(ref_outputs, test_outputs)
        diff_records.append(diff)

        if not args.skip_output_save:
            save_sample_npz(samples_dir / f"sample_{i:04d}_inputs.npz", inputs)
            save_sample_npz(samples_dir / f"sample_{i:04d}_ref_outputs.npz", ref_outputs)
            save_sample_npz(samples_dir / f"sample_{i:04d}_test_outputs.npz", test_outputs)

    aggregated = aggregate_diffs(diff_records)

    report: Dict[str, Any] = {
        "reference_type": reference_type,
        "test_engine": str(Path(args.test_engine).resolve()),
        "batch": args.batch,
        "samples": args.n,
        "seed": args.seed,
        "input_shapes": {name: list(shape) for name, shape in ref_shapes.items()},
        "diff": aggregated,
    }

    if reference_type == "tensorrt":
        report["ref_engine"] = str(reference_path)
    else:
        report["ref_torch"] = str(reference_path)
        report["torch_device"] = torch_device
        report["torch_input_style"] = args.torch_input_style

    summary_path = args.outdir / "output_diff.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    per_sample_path = args.outdir / "per_sample_diff.json"
    with per_sample_path.open("w", encoding="utf-8") as f:
        json.dump(diff_records, f, indent=2)

    print(f"Saved comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
