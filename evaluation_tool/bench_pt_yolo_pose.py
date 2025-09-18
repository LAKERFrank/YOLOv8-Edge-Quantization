#!/usr/bin/env python3
"""Benchmark PyTorch YOLOv8-Pose models on GPU inference only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Sequence

import torch


@dataclass
class BenchmarkArgs:
    pt: str
    imgsz: int
    ch: int
    batch: int
    iters: int
    warmup: int
    device: str
    dtype: str
    no_tf32: bool
    ultra: bool


@dataclass
class BenchmarkResult:
    throughput_qps: float
    total_host_walltime_s: float
    metrics: Dict[str, Dict[str, float]]


CSV_OUTPUT_PATH = "bench_pt_result.csv"
JSON_OUTPUT_PATH = "bench_pt_result.json"
SEED = 2023
SUPPORTED_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def parse_args(argv: Optional[Sequence[str]] = None) -> BenchmarkArgs:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv8 Pose PyTorch models on GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pt", required=True, type=str, help="Path to YOLOv8-Pose .pt weights")
    parser.add_argument("--imgsz", default=640, type=int, help="Input image size (square)")
    parser.add_argument("--ch", default=1, type=int, choices=(1, 3), help="Number of input channels")
    parser.add_argument("--batch", default=1, type=int, help="Batch size")
    parser.add_argument("--iters", default=2000, type=int, help="Number of timed iterations")
    parser.add_argument("--warmup", default=200, type=int, help="Number of warmup iterations")
    parser.add_argument("--device", default="cuda:0", type=str, help="CUDA device string")
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=tuple(SUPPORTED_DTYPES.keys()),
        help="Computation dtype",
    )
    parser.add_argument("--no_tf32", action="store_true", help="Disable TF32 computation")
    parser.add_argument("--ultra", action="store_true", help="Use ultralytics.YOLO loader first")
    args = parser.parse_args(argv)

    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    return BenchmarkArgs(
        pt=args.pt,
        imgsz=args.imgsz,
        ch=args.ch,
        batch=args.batch,
        iters=args.iters,
        warmup=args.warmup,
        device=args.device,
        dtype=args.dtype,
        no_tf32=args.no_tf32,
        ultra=args.ultra,
    )


def setup_environment(device: str, allow_tf32: bool) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for benchmarking")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32

    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)
    return torch_device


def load_model(pt_path: str, device: torch.device, prefer_ultra: bool) -> torch.nn.Module:
    module: Optional[torch.nn.Module] = None
    errors: List[str] = []

    if prefer_ultra:
        try:
            from ultralytics import YOLO  # type: ignore

            yolo_model = YOLO(pt_path)
            if hasattr(yolo_model, "model") and isinstance(yolo_model.model, torch.nn.Module):
                module = yolo_model.model
            elif isinstance(yolo_model, torch.nn.Module):
                module = yolo_model
            else:
                errors.append("ultralytics.YOLO did not return an nn.Module")
        except Exception as exc:  # pragma: no cover - optional dependency
            errors.append(f"ultralytics.YOLO load failed: {exc}")

    if module is None:
        loaded = torch.load(pt_path, map_location=device)
        module = extract_module(loaded)
        if module is None:
            errors.append("torch.load() did not return an nn.Module or contain a model")

    if module is None:
        raise RuntimeError(
            "Failed to load model from {}. Errors: {}".format(pt_path, "; ".join(errors) or "unknown")
        )

    module.to(device)
    module.eval()
    return module


def extract_module(obj: Any) -> Optional[torch.nn.Module]:
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, MutableMapping):
        for key in ("model", "ema"):
            candidate = obj.get(key)
            if isinstance(candidate, torch.nn.Module):
                return candidate
    if hasattr(obj, "model") and isinstance(obj.model, torch.nn.Module):
        return obj.model
    return None


def allocate_host_input(args: BenchmarkArgs) -> torch.Tensor:
    shape = (args.batch, args.ch, args.imgsz, args.imgsz)
    tensor = torch.randn(shape, dtype=torch.float32)
    return tensor.pin_memory()


def create_device_buffer(host_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.empty_like(host_tensor, device=device)


def autocast_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def compute_percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_latency(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: float("nan") for k in ("min", "max", "mean", "median", "p90", "p95", "p99")}
    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    summary = {
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": total / len(sorted_vals),
        "median": compute_percentile(sorted_vals, 50),
        "p90": compute_percentile(sorted_vals, 90),
        "p95": compute_percentile(sorted_vals, 95),
        "p99": compute_percentile(sorted_vals, 99),
    }
    return summary


def iter_tensors(output: Any, visited: Optional[set[int]] = None) -> Iterator[torch.Tensor]:
    if visited is None:
        visited = set()
    obj_id = id(output)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(output, torch.Tensor):
        yield output
        return
    if isinstance(output, (list, tuple, set)):
        for item in output:
            yield from iter_tensors(item, visited)
        return
    if isinstance(output, dict):
        for value in output.values():
            yield from iter_tensors(value, visited)
        return

    for attr in ("tensor", "data", "output", "pred", "boxes", "keypoints", "heatmaps"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            yield from iter_tensors(value, visited)


def select_largest_tensor(output: Any) -> torch.Tensor:
    largest: Optional[torch.Tensor] = None
    for tensor in iter_tensors(output):
        if tensor is None:
            continue
        if largest is None or tensor.numel() > largest.numel():
            largest = tensor
    if largest is None:
        raise RuntimeError("Model output does not contain any torch.Tensor objects")
    return largest


def ensure_pinned_buffer(buffer: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if buffer is not None:
        if buffer.shape == reference.shape and buffer.dtype == reference.dtype:
            return buffer
    new_buffer = torch.empty(reference.shape, dtype=reference.dtype, device="cpu")
    return new_buffer.pin_memory()


def benchmark(args: BenchmarkArgs) -> BenchmarkResult:
    allow_tf32 = not args.no_tf32
    device = setup_environment(args.device, allow_tf32)
    dtype = SUPPORTED_DTYPES[args.dtype]

    model = load_model(args.pt, device, args.ultra)

    host_input = allocate_host_input(args)
    device_input = create_device_buffer(host_input, device)

    h2d_times: List[float] = []
    gpu_times: List[float] = []
    d2h_times: List[float] = []
    host_times: List[float] = []

    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    d2h_start = torch.cuda.Event(enable_timing=True)
    d2h_end = torch.cuda.Event(enable_timing=True)

    pinned_output: Optional[torch.Tensor] = None

    total_iterations = args.warmup + args.iters

    autocast_enabled = dtype == torch.float16

    with torch.inference_mode():
        for iteration in range(total_iterations):
            record = iteration >= args.warmup

            host_t0 = time.perf_counter()

            h2d_start.record()
            device_input.copy_(host_input, non_blocking=True)
            h2d_end.record()
            h2d_end.synchronize()
            if record:
                h2d_times.append(h2d_start.elapsed_time(h2d_end))

            gpu_start.record()
            with autocast_context(autocast_enabled):
                output = model(device_input)
            gpu_end.record()
            gpu_end.synchronize()
            if record:
                gpu_times.append(gpu_start.elapsed_time(gpu_end))

            largest_tensor = select_largest_tensor(output)
            pinned_output = ensure_pinned_buffer(pinned_output, largest_tensor)

            d2h_start.record()
            pinned_output.copy_(largest_tensor, non_blocking=True)
            d2h_end.record()
            d2h_end.synchronize()
            if record:
                d2h_times.append(d2h_start.elapsed_time(d2h_end))

            torch.cuda.synchronize()
            host_elapsed_ms = (time.perf_counter() - host_t0) * 1000.0
            if record:
                host_times.append(host_elapsed_ms)

    total_host_walltime_s = sum(host_times) / 1000.0 if host_times else 0.0
    throughput_qps = (args.iters / total_host_walltime_s) if total_host_walltime_s > 0 else 0.0

    metrics = {
        "host": summarize_latency(host_times),
        "h2d": summarize_latency(h2d_times),
        "gpu": summarize_latency(gpu_times),
        "d2h": summarize_latency(d2h_times),
    }

    return BenchmarkResult(
        throughput_qps=throughput_qps,
        total_host_walltime_s=total_host_walltime_s,
        metrics=metrics,
    )


def print_report(args: BenchmarkArgs, result: BenchmarkResult, allow_tf32: bool) -> None:
    input_shape = [args.batch, args.ch, args.imgsz, args.imgsz]

    print("Model")
    print("-----")
    print(f"Path: {args.pt}")
    print(f"Device: {args.device}")
    print(f"Input shape: {input_shape}")
    print(f"Dtype: {args.dtype}")
    print(f"TF32: {'On' if allow_tf32 else 'Off'}")
    print()

    print("Performance summary")
    print("-------------------")
    print(f"Throughput (qps): {result.throughput_qps:.4f}")
    print(f"Total Host Walltime (s): {result.total_host_walltime_s:.6f}")
    print()

    print("Latency breakdown (ms)")
    print("----------------------")
    headers = ("Latency", "min", "max", "mean", "median", "p90", "p95", "p99")
    row_format = "{:<10} " + " ".join(["{:>10}"] * (len(headers) - 1))
    print(row_format.format(*headers))
    for key in ("host", "h2d", "gpu", "d2h"):
        stats = result.metrics[key]
        values = [
            key,
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
            f"{stats['mean']:.4f}",
            f"{stats['median']:.4f}",
            f"{stats['p90']:.4f}",
            f"{stats['p95']:.4f}",
            f"{stats['p99']:.4f}",
        ]
        print(row_format.format(*values))


def write_csv(result: BenchmarkResult, output_path: str = CSV_OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    headers = ["metric", "min", "max", "mean", "median", "p90", "p95", "p99"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for metric in ("host", "h2d", "gpu", "d2h"):
            stats = result.metrics[metric]
            writer.writerow(
                [
                    metric,
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}",
                    f"{stats['mean']:.6f}",
                    f"{stats['median']:.6f}",
                    f"{stats['p90']:.6f}",
                    f"{stats['p95']:.6f}",
                    f"{stats['p99']:.6f}",
                ]
            )


def write_json(args: BenchmarkArgs, result: BenchmarkResult, allow_tf32: bool, output_path: str = JSON_OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    payload = {
        "model_path": os.path.abspath(args.pt),
        "device": args.device,
        "batch": args.batch,
        "shape": [args.batch, args.ch, args.imgsz, args.imgsz],
        "dtype": args.dtype,
        "tf32": allow_tf32,
        "iters": args.iters,
        "warmup": args.warmup,
        "throughput_qps": result.throughput_qps,
        "total_host_walltime_s": result.total_host_walltime_s,
        "metrics": result.metrics,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    allow_tf32 = not args.no_tf32

    result = benchmark(args)
    print_report(args, result, allow_tf32)
    write_csv(result)
    write_json(args, result, allow_tf32)


if __name__ == "__main__":
    main()
