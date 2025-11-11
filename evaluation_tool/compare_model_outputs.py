#!/usr/bin/env python3
"""Compare YOLOv8 pose performance between PyTorch and TensorRT benchmarks."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import math
import os
import re
import shlex
import subprocess
import sys
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bench_pt_yolo_pose  # noqa: E402

STAT_NAMES: Tuple[str, ...] = ("min", "max", "mean", "median", "p90", "p95", "p99")
DEFAULT_ARTIFACT_DIR = Path("artifacts/compare")


@dataclass
class BenchmarkSummary:
    """Normalized view over latency statistics for a single benchmark run."""

    label: str
    path: str
    dtype: Optional[str]
    tf32: Optional[bool]
    throughput_qps: Optional[float]
    total_host_walltime_s: Optional[float]
    latencies: Dict[str, Dict[str, float]]
    input_spec: Optional[str] = None


@dataclass
class ComparisonRow:
    """Container for a single comparison entry."""

    category: str
    metric: str
    pt_value: Optional[float]
    engine_value: Optional[float]
    delta: Optional[float]
    ratio: Optional[float]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a PyTorch YOLOv8 pose model against a TensorRT engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pt", required=True, type=str, help="Path to YOLOv8 pose .pt weights")
    parser.add_argument("--engine", required=True, type=str, help="Path to TensorRT engine file")
    parser.add_argument("--imgsz", default=640, type=int, help="Input image resolution (square)")
    parser.add_argument("--ch", default=1, type=int, help="Number of model input channels")
    parser.add_argument("--batch", default=1, type=int, help="Batch size for benchmarking")
    parser.add_argument("--pt-iters", default=2000, type=int, help="Timed iterations for PyTorch benchmark")
    parser.add_argument("--pt-warmup", default=200, type=int, help="Warmup iterations for PyTorch benchmark")
    parser.add_argument(
        "--pt-dtype",
        default="fp32",
        type=str,
        choices=tuple(bench_pt_yolo_pose.SUPPORTED_DTYPES.keys()),
        help="Computation precision for the PyTorch benchmark",
    )
    parser.add_argument("--pt-device", default="cuda:0", type=str, help="CUDA device string")
    parser.add_argument("--pt-no-tf32", action="store_true", help="Disable TF32 for PyTorch benchmark")
    parser.add_argument("--pt-ultra", action="store_true", help="Attempt ultralytics.YOLO loader first")
    parser.add_argument(
        "--shapes",
        type=str,
        help="TensorRT input shapes definition (e.g. images:1x1x640x640). Defaults to batch/ch/imgsz args.",
    )
    parser.add_argument("--trtexec", default="trtexec", type=str, help="trtexec binary to execute")
    parser.add_argument(
        "--trtexec-args",
        default="",
        type=str,
        help="Additional arguments appended to trtexec invocation (quoted string)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_ARTIFACT_DIR,
        type=Path,
        help="Directory to store comparison CSV output",
    )

    args = parser.parse_args(argv)

    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.ch <= 0:
        parser.error("--ch must be positive")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.pt_iters <= 0:
        parser.error("--pt-iters must be positive")
    if args.pt_warmup < 0:
        parser.error("--pt-warmup must be non-negative")

    return args


def _collect_ultralytics_classes() -> List[type]:
    """Collect Ultralytics classes needed for safe torch.load allowlisting."""

    try:
        import ultralytics  # type: ignore
    except ImportError:
        return []

    module_names = {"ultralytics"}

    if hasattr(ultralytics, "__path__"):
        for module_info in pkgutil.walk_packages(ultralytics.__path__, ultralytics.__name__ + "."):
            module_names.add(module_info.name)

    classes: List[type] = []
    seen: set[type] = set()
    for module_name in sorted(module_names):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj in seen:
                continue
            obj_module = getattr(obj, "__module__", "")
            if obj_module.startswith("ultralytics"):
                classes.append(obj)
                seen.add(obj)

    return classes


def maybe_allow_ultralytics_safe_globals() -> None:
    """Allowlist Ultralytics and container classes for PyTorch checkpoint deserialization."""

    serialization_spec = importlib.util.find_spec("torch.serialization")
    if serialization_spec is None:
        return

    serialization_module = importlib.import_module("torch.serialization")
    add_safe_globals = getattr(serialization_module, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    safe_classes: List[type] = []
    safe_classes.extend(_collect_ultralytics_classes())

    container_spec = importlib.util.find_spec("torch.nn.modules.container")
    if container_spec is not None:
        container_module = importlib.import_module("torch.nn.modules.container")
        for attr in ("Sequential", "ModuleList", "ParameterList"):
            container_cls = getattr(container_module, attr, None)
            if container_cls is not None:
                safe_classes.append(container_cls)

    if safe_classes:
        add_safe_globals(safe_classes)


def run_pt_benchmark(args: argparse.Namespace) -> BenchmarkSummary:
    """Execute the PyTorch benchmark via bench_pt_yolo_pose.py."""

    maybe_allow_ultralytics_safe_globals()
    bench_args = bench_pt_yolo_pose.BenchmarkArgs(
        pt=args.pt,
        imgsz=args.imgsz,
        ch=args.ch,
        batch=args.batch,
        iters=args.pt_iters,
        warmup=args.pt_warmup,
        device=args.pt_device,
        dtype=args.pt_dtype,
        no_tf32=args.pt_no_tf32,
        ultra=args.pt_ultra,
    )

    result = bench_pt_yolo_pose.benchmark(bench_args)
    latencies = {
        name: {stat: float(value) for stat, value in stats.items()}
        for name, stats in result.metrics.items()
    }

    input_spec = f"{args.batch}x{args.ch}x{args.imgsz}x{args.imgsz}"

    return BenchmarkSummary(
        label="PyTorch",
        path=os.path.abspath(args.pt),
        dtype=args.pt_dtype,
        tf32=not args.pt_no_tf32,
        throughput_qps=result.throughput_qps,
        total_host_walltime_s=result.total_host_walltime_s,
        latencies=latencies,
        input_spec=input_spec,
    )


def build_trtexec_command(args: argparse.Namespace, shapes: str) -> List[str]:
    base_cmd = [
        args.trtexec,
        f"--loadEngine={args.engine}",
        f"--shapes={shapes}",
        "--verbose",
    ]
    extra = shlex.split(args.trtexec_args) if args.trtexec_args else []
    return base_cmd + extra


def run_trtexec(args: argparse.Namespace, shapes: str) -> Tuple[BenchmarkSummary, str]:
    """Execute trtexec and parse its verbose summary."""

    cmd = build_trtexec_command(args, shapes)
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "trtexec command failed with exit code {}\nCommand: {}\nOutput:\n{}".format(
                completed.returncode, " ".join(cmd), completed.stdout
            )
        )

    summary = parse_trtexec_output(completed.stdout)
    summary.label = "TensorRT"
    summary.path = os.path.abspath(args.engine)
    summary.input_spec = shapes
    return summary, completed.stdout


def parse_trtexec_output(output: str) -> BenchmarkSummary:
    """Parse throughput, TF32 and latency tables from trtexec output."""

    dtype: Optional[str] = None
    tf32: Optional[bool] = None
    throughput: Optional[float] = None
    walltime: Optional[float] = None
    latencies: Dict[str, Dict[str, float]] = {}

    lines = output.splitlines()

    dtype_pattern = re.compile(r"dtype\s*[:=]\s*([A-Za-z0-9_]+)", re.IGNORECASE)
    tf32_pattern = re.compile(r"TF32\s*[:=]\s*(on|off|enabled|disabled)", re.IGNORECASE)
    throughput_pattern = re.compile(r"Throughput\s*[:=]\s*([0-9.+-eE]+)")
    walltime_pattern = re.compile(r"Total\s+Host\s+Walltime[^0-9]*([0-9.+-eE]+)")
    latency_header_pattern = re.compile(
        r"Latency\s+min\s+max\s+mean\s+median\s+p90\s+p95\s+p99",
        re.IGNORECASE,
    )

    header_index: Optional[int] = None
    for idx, line in enumerate(lines):
        if dtype is None:
            dtype_match = dtype_pattern.search(line)
            if dtype_match:
                dtype = dtype_match.group(1).lower()
        if tf32 is None:
            tf32_match = tf32_pattern.search(line)
            if tf32_match:
                tf32_value = tf32_match.group(1).lower()
                tf32 = tf32_value in {"on", "enabled", "true"}
        if throughput is None:
            throughput_match = throughput_pattern.search(line)
            if throughput_match:
                throughput = float(throughput_match.group(1))
        if walltime is None:
            walltime_match = walltime_pattern.search(line)
            if walltime_match:
                walltime = float(walltime_match.group(1))
        if header_index is None and latency_header_pattern.search(line):
            header_index = idx

    if header_index is not None:
        for line in lines[header_index + 1 :]:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("===") or stripped.startswith("---"):
                break
            parts = stripped.split()
            if len(parts) < len(STAT_NAMES) + 1:
                continue
            key = parts[0].lower()
            try:
                values = [float(value) for value in parts[1 : len(STAT_NAMES) + 1]]
            except ValueError:
                continue
            latencies[key] = dict(zip(STAT_NAMES, values))

    if not latencies:
        inline_pattern = re.compile(
            r"([A-Za-z0-9_/]+)\s+latency\s*[:=]\s*min\s*=\s*([0-9.+-eE]+).*?max\s*=\s*([0-9.+-eE]+).*?"
            r"mean\s*=\s*([0-9.+-eE]+).*?median\s*=\s*([0-9.+-eE]+).*?percentile\(90%\)\s*=\s*([0-9.+-eE]+).*?"
            r"percentile\(95%\)\s*=\s*([0-9.+-eE]+).*?percentile\(99%\)\s*=\s*([0-9.+-eE]+)",
            re.IGNORECASE,
        )
        for line in lines:
            match = inline_pattern.search(line)
            if not match:
                continue
            key = match.group(1).lower()
            values = [float(match.group(i)) for i in range(2, 9)]
            latencies[key] = dict(zip(STAT_NAMES, values))

    return BenchmarkSummary(
        label="TensorRT",
        path="",
        dtype=dtype,
        tf32=tf32,
        throughput_qps=throughput,
        total_host_walltime_s=walltime,
        latencies=latencies,
    )


def format_float(value: Optional[float], precision: int = 4) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def format_ratio(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def get_latency(summary: BenchmarkSummary, component: str, stat: str) -> Optional[float]:
    component_stats = summary.latencies.get(component)
    if component_stats is None:
        return None
    value = component_stats.get(stat)
    return float(value) if value is not None else None


def merge_latency_components(
    pt_summary: BenchmarkSummary, engine_summary: BenchmarkSummary
) -> List[str]:
    """Merge latency component keys while preserving their original ordering."""

    ordered_components: List[str] = []
    seen = set()

    for key in pt_summary.latencies.keys():
        if key not in seen:
            ordered_components.append(key)
            seen.add(key)

    for key in engine_summary.latencies.keys():
        if key not in seen:
            ordered_components.append(key)
            seen.add(key)

    return ordered_components


def build_comparison_rows(pt_summary: BenchmarkSummary, engine_summary: BenchmarkSummary) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []

    rows.append(make_comparison_row("summary", "throughput_qps", pt_summary.throughput_qps, engine_summary.throughput_qps))
    rows.append(
        make_comparison_row(
            "summary",
            "total_host_walltime_s",
            pt_summary.total_host_walltime_s,
            engine_summary.total_host_walltime_s,
        )
    )

    components = merge_latency_components(pt_summary, engine_summary)
    for component in components:
        for stat in STAT_NAMES:
            rows.append(
                make_comparison_row(
                    "latency",
                    f"{component}.{stat}",
                    get_latency(pt_summary, component, stat),
                    get_latency(engine_summary, component, stat),
                )
            )

    return rows


def make_comparison_row(
    category: str,
    metric: str,
    pt_value: Optional[float],
    engine_value: Optional[float],
) -> ComparisonRow:
    delta: Optional[float] = None
    ratio: Optional[float] = None

    if pt_value is not None and engine_value is not None:
        if not math.isnan(pt_value) and not math.isnan(engine_value):
            delta = engine_value - pt_value
            if pt_value != 0.0:
                ratio = engine_value / pt_value

    return ComparisonRow(category, metric, pt_value, engine_value, delta, ratio)


def write_comparison_csv(output_path: Path, rows: Iterable[ComparisonRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "metric", "pt_value", "engine_value", "delta", "ratio"])
        for row in rows:
            writer.writerow(
                [
                    row.category,
                    row.metric,
                    format_csv_value(row.pt_value),
                    format_csv_value(row.engine_value),
                    format_csv_value(row.delta),
                    format_csv_value(row.ratio),
                ]
            )


def format_csv_value(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_output_path(output_dir: Path, pt_path: str, engine_path: str) -> Path:
    pt_name = sanitize_name(Path(pt_path).stem)
    engine_name = sanitize_name(Path(engine_path).stem)
    filename = f"{pt_name}__vs__{engine_name}.csv"
    return output_dir / filename


def print_benchmark_summary(summary: BenchmarkSummary) -> None:
    print(summary.label)
    print("-" * len(summary.label))
    print(f"Path: {summary.path}")
    if summary.input_spec:
        print(f"Input: {summary.input_spec}")
    if summary.dtype:
        print(f"Dtype: {summary.dtype}")
    if summary.tf32 is not None:
        print(f"TF32: {'On' if summary.tf32 else 'Off'}")
    print()
    print("Performance summary")
    print("-------------------")
    print(f"Throughput (qps): {format_float(summary.throughput_qps, precision=4)}")
    print(f"Total Host Walltime (s): {format_float(summary.total_host_walltime_s, precision=6)}")
    print()
    if summary.latencies:
        print("Latency breakdown (ms)")
        print("----------------------")
        header = ("Latency",) + STAT_NAMES
        row_format = "{:<10} " + " ".join(["{:>10}"] * len(STAT_NAMES))
        print(row_format.format(*header))
        for key, stats in summary.latencies.items():
            values = [format_float(stats.get(stat), precision=4) for stat in STAT_NAMES]
            print(row_format.format(key, *values))
    print()


def print_comparison_table(rows: List[ComparisonRow]) -> None:
    summary_rows = [row for row in rows if row.category == "summary"]
    latency_rows = [row for row in rows if row.category == "latency"]

    print("Comparison (TensorRT vs PyTorch)")
    print("-------------------------------")
    header = ("Metric", "PyTorch", "TensorRT", "Delta", "Ratio")
    row_format = "{:<28} " + " ".join(["{:>12}"] * (len(header) - 1))
    print(row_format.format(*header))
    for row in summary_rows:
        pretty_metric = {
            "throughput_qps": "Throughput (qps)",
            "total_host_walltime_s": "Total Host Walltime (s)",
        }.get(row.metric, row.metric)
        print(
            row_format.format(
                pretty_metric,
                format_float(row.pt_value, precision=4),
                format_float(row.engine_value, precision=4),
                format_float(row.delta, precision=4),
                format_ratio(row.ratio),
            )
        )

    if latency_rows:
        print()
        print("Latency deltas (TensorRT - PyTorch)")
        print("-----------------------------------")
        print(row_format.format(*header))
        for row in latency_rows:
            print(
                row_format.format(
                    row.metric,
                    format_float(row.pt_value, precision=4),
                    format_float(row.engine_value, precision=4),
                    format_float(row.delta, precision=4),
                    format_ratio(row.ratio),
                )
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    shapes = args.shapes or f"images:{args.batch}x{args.ch}x{args.imgsz}x{args.imgsz}"

    print("Running PyTorch benchmark...")
    pt_summary = run_pt_benchmark(args)
    print()
    print_benchmark_summary(pt_summary)

    print("Running TensorRT benchmark...")
    engine_summary, _ = run_trtexec(args, shapes)
    print()
    print_benchmark_summary(engine_summary)

    rows = build_comparison_rows(pt_summary, engine_summary)
    print_comparison_table(rows)

    output_path = build_output_path(args.output_dir, args.pt, args.engine)
    write_comparison_csv(output_path, rows)
    print()
    print(f"Comparison CSV written to: {output_path}")


if __name__ == "__main__":
    main()
