#!/usr/bin/env python3
"""Compare TensorRT engine outputs between a reference and a test engine."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
    info: Dict[str, BindingInfo] = {}
    for idx in range(engine.num_bindings):
        name = engine.get_binding_name(idx)
        dtype = np.dtype(trt.nptype(engine.get_binding_dtype(idx)))
        info[name] = BindingInfo(
            name=name,
            index=idx,
            dtype=dtype,
            is_input=engine.binding_is_input(idx),
        )
    return info


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
    context = engine.create_execution_context()
    if engine.num_optimization_profiles > 0:
        try:
            context.set_optimization_profile(profile_index)
        except AttributeError:
            context.active_optimization_profile = profile_index
    shapes: Dict[str, Tuple[int, ...]] = {}
    for idx in range(engine.num_bindings):
        if not engine.binding_is_input(idx):
            continue
        name = engine.get_binding_name(idx)
        if name in provided:
            shapes[name] = tuple(provided[name])
            continue
        dims = tuple(context.get_binding_shape(idx))
        if engine.has_implicit_batch_dimension:
            dims = (batch_size, *tuple(int(d) for d in engine.get_binding_shape(idx)))
        if -1 in dims:
            opt_shape: Optional[Tuple[int, ...]] = None
            if hasattr(engine, "get_profile_shape"):
                try:
                    profile_shapes = engine.get_profile_shape(profile_index, idx)
                except TypeError:
                    profile_shapes = None
                if profile_shapes:
                    # profile_shapes is (min, opt, max)
                    try:
                        opt_shape = tuple(int(d) for d in profile_shapes[1])
                    except TypeError:
                        opt_shape = None
            if opt_shape is not None and -1 not in opt_shape:
                dims = opt_shape
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
        self.context = engine.create_execution_context()
        if engine.num_optimization_profiles > 0:
            try:
                self.context.set_optimization_profile(profile_index)
            except AttributeError:
                self.context.active_optimization_profile = profile_index
        self.stream = cuda.Stream()

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        bindings: List[int] = [0] * self.engine.num_bindings
        device_allocations: List[cuda.DeviceAllocation] = []
        host_outputs: Dict[str, np.ndarray] = {}
        output_allocations: Dict[str, cuda.DeviceAllocation] = {}

        prepared_inputs: Dict[str, np.ndarray] = {}
        for idx in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(idx):
                continue
            name = self.engine.get_binding_name(idx)
            if name not in inputs:
                raise KeyError(f"Missing input '{name}' for inference")
            arr = np.asarray(inputs[name])
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(idx)))
            if arr.dtype != dtype:
                arr = arr.astype(dtype)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            if self.implicit_batch:
                expected_dims = tuple(self.engine.get_binding_shape(idx))
                expected_shape = (self.batch_size, *tuple(int(d) for d in expected_dims))
                if tuple(arr.shape) != expected_shape:
                    raise ValueError(
                        f"Input '{name}' expected shape {expected_shape} for implicit batch engine, got {arr.shape}"
                    )
            prepared_inputs[name] = arr
            if not self.implicit_batch:
                self.context.set_binding_shape(idx, arr.shape)

        if not self.implicit_batch and hasattr(self.context, "all_binding_shapes_specified"):
            if not self.context.all_binding_shapes_specified:
                missing = [
                    self.engine.get_binding_name(i)
                    for i in range(self.engine.num_bindings)
                    if self.engine.binding_is_input(i)
                    and -1 in tuple(self.context.get_binding_shape(i))
                ]
                raise RuntimeError(f"Binding shapes not fully specified: {missing}")

        # Allocate buffers and copy inputs
        for idx in range(self.engine.num_bindings):
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(idx)))
            if self.engine.binding_is_input(idx):
                name = self.engine.get_binding_name(idx)
                arr = prepared_inputs[name]
                device_mem = cuda.mem_alloc(arr.nbytes)
                device_allocations.append(device_mem)
                cuda.memcpy_htod_async(device_mem, arr, self.stream)
                bindings[idx] = int(device_mem)
            else:
                if self.implicit_batch:
                    dims = tuple(self.engine.get_binding_shape(idx))
                    out_shape = (self.batch_size, *tuple(int(d) for d in dims))
                else:
                    out_shape = tuple(self.context.get_binding_shape(idx))
                if -1 in out_shape:
                    raise RuntimeError(
                        f"Output binding '{self.engine.get_binding_name(idx)}' has unresolved shape {out_shape}"
                    )
                host_array = np.empty(out_shape, dtype=dtype)
                device_mem = cuda.mem_alloc(host_array.nbytes)
                device_allocations.append(device_mem)
                bindings[idx] = int(device_mem)
                name = self.engine.get_binding_name(idx)
                host_outputs[name] = host_array
                output_allocations[name] = device_mem

        # Execute
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
    parser = argparse.ArgumentParser(description="Compare outputs of two TensorRT engines.")
    parser.add_argument("--ref_engine", required=True, type=Path, help="Reference (e.g., FP32) engine")
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
    args = parser.parse_args()

    provided_shapes: Dict[str, Tuple[int, ...]] = {}
    for entry in args.input_shape:
        name, dims = parse_shape_string(entry)
        provided_shapes[name] = dims

    args.outdir.mkdir(parents=True, exist_ok=True)

    ref_engine = load_engine(args.ref_engine)
    test_engine = load_engine(args.test_engine)

    ref_bindings = build_binding_info(ref_engine)
    test_bindings = build_binding_info(test_engine)
    ensure_engines_compatible(ref_bindings, test_bindings)

    ref_shapes = resolve_input_shapes(ref_engine, provided_shapes, args.batch, args.profile_index)
    test_shapes = resolve_input_shapes(test_engine, provided_shapes, args.batch, args.profile_index)
    for name, shape in ref_shapes.items():
        if name not in test_shapes:
            raise ValueError(f"Input '{name}' missing in test engine")
        if tuple(test_shapes[name]) != tuple(shape):
            raise ValueError(
                f"Input shape mismatch for '{name}': {shape} vs {test_shapes[name]}"
            )

    rng = np.random.default_rng(args.seed)

    ref_runner = TRTEngineRunner(ref_engine, args.batch, args.profile_index)
    test_runner = TRTEngineRunner(test_engine, args.batch, args.profile_index)

    diff_records: List[Dict[str, Mapping[str, float]]] = []
    samples_dir = args.outdir / "samples"
    if not args.skip_output_save:
        samples_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        inputs = generate_random_inputs(ref_bindings, ref_shapes, rng)
        ref_outputs = ref_runner.infer(inputs)
        test_outputs = test_runner.infer(inputs)
        diff = compute_diff(ref_outputs, test_outputs)
        diff_records.append(diff)

        if not args.skip_output_save:
            save_sample_npz(samples_dir / f"sample_{i:04d}_inputs.npz", inputs)
            save_sample_npz(samples_dir / f"sample_{i:04d}_ref_outputs.npz", ref_outputs)
            save_sample_npz(samples_dir / f"sample_{i:04d}_test_outputs.npz", test_outputs)

    aggregated = aggregate_diffs(diff_records)

    report = {
        "ref_engine": str(Path(args.ref_engine).resolve()),
        "test_engine": str(Path(args.test_engine).resolve()),
        "batch": args.batch,
        "samples": args.n,
        "seed": args.seed,
        "input_shapes": {name: list(shape) for name, shape in ref_shapes.items()},
        "diff": aggregated,
    }

    summary_path = args.outdir / "output_diff.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    per_sample_path = args.outdir / "per_sample_diff.json"
    with per_sample_path.open("w", encoding="utf-8") as f:
        json.dump(diff_records, f, indent=2)

    print(f"Saved comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
