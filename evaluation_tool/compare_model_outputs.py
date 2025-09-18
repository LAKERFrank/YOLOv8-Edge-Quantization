#!/usr/bin/env python3
"""Benchmark PyTorch models against TensorRT engines and compare performance."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from bench_pt_yolo_pose import (
    BenchmarkArgs as PTBenchmarkArgs,
    BenchmarkResult as PTBenchmarkResult,
    SUPPORTED_DTYPES as PT_SUPPORTED_DTYPES,
    benchmark as run_pt_benchmark,
)


def allow_ultralytics_pose_pickles() -> None:
    """Allowlist Ultralytics PoseModel so torch.load works with weights_only checkpoints."""

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - import depends on environment
        return

    serialization = getattr(torch, "serialization", None)
    if serialization is None:
        return

    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    try:
        from ultralytics.nn.tasks import PoseModel  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    try:
        add_safe_globals([PoseModel])
    except Exception:  # pragma: no cover - safety guard only
        return

STAT_NAMES = ("min", "max", "mean", "median", "p90", "p95", "p99")
METRIC_NAMES = ("host", "h2d", "gpu", "d2h")
LABEL_TO_METRIC = (
    ("host latency", "host"),
    ("latency", "host"),
    ("h2d latency", "h2d"),
    ("gpu compute time", "gpu"),
    ("gpu latency", "gpu"),
    ("d2h latency", "d2h"),
)


@dataclass
class ModelSummary:
    title: str
    path: str
    dtype: Optional[str]
    tf32: Optional[bool]
    throughput_qps: float
    total_host_walltime_s: float
    metrics: Dict[str, Dict[str, float]]


@dataclass
class TrtexecParseResult:
    throughput_qps: float
    total_host_walltime_s: float
    metrics: Dict[str, Dict[str, float]]
    dtype: Optional[str]
    tf32: Optional[bool]
    raw_output: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a PyTorch .pt model and a TensorRT engine, then compare the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pt", required=True, help="Path to the YOLOv8-Pose .pt weights")
    parser.add_argument("--engine", required=True, help="Path to the TensorRT engine file")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for benchmarking")
    parser.add_argument("--ch", type=int, default=1, choices=(1, 3), help="Number of input channels")
    parser.add_argument("--pt-iters", type=int, default=2000, help="Number of timed iterations for PyTorch")
    parser.add_argument("--pt-warmup", type=int, default=200, help="Number of warmup iterations for PyTorch")
    parser.add_argument("--pt-device", type=str, default="cuda:0", help="CUDA device string for PyTorch")
    parser.add_argument(
        "--pt-dtype",
        type=str,
        default="fp32",
        choices=tuple(PT_SUPPORTED_DTYPES.keys()),
        help="Computation dtype for PyTorch benchmarking",
    )
    parser.add_argument("--pt-no-tf32", action="store_true", help="Disable TF32 for PyTorch benchmarking")
    parser.add_argument("--pt-ultra", action="store_true", help="Use ultralytics.YOLO loader for the PyTorch model")
    parser.add_argument("--input-name", type=str, default="images", help="Input tensor name for the TensorRT engine")
    parser.add_argument("--shapes", type=str, default=None, help="Explicit shapes string passed to trtexec")
    parser.add_argument("--trtexec", type=str, default="trtexec", help="Path to the trtexec binary")
    parser.add_argument(
        "--trtexec-extra-args",
        type=str,
        default="",
        help="Additional arguments for trtexec (provide as a quoted string)",
    )
    parser.add_argument("--no-trtexec-verbose", action="store_true", help="Do not pass --verbose to trtexec")
    parser.add_argument(
        "--engine-dtype",
        type=str,
        default="auto",
        help="Override the dtype label for the engine. Use 'auto' to rely on trtexec output.",
    )
    parser.add_argument(
        "--engine-tf32",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help="Report TF32 status for the engine output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/compare"),
        help="Directory where the comparison CSV will be written",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional name for the comparison CSV file",
    )
    args = parser.parse_args(argv)

    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.pt_iters <= 0:
        parser.error("--pt-iters must be positive")
    if args.pt_warmup < 0:
        parser.error("--pt-warmup must be non-negative")

    return args


def run_pt_evaluation(args: argparse.Namespace) -> ModelSummary:
    allow_ultralytics_pose_pickles()

    pt_path = Path(args.pt).expanduser().resolve()
    pt_args = PTBenchmarkArgs(
        pt=str(pt_path),
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
    result: PTBenchmarkResult = run_pt_benchmark(pt_args)
    metrics = normalise_metrics(result.metrics)
    tf32_status = not args.pt_no_tf32

    return ModelSummary(
        title="PyTorch Model",
        path=str(pt_path),
        dtype=args.pt_dtype,
        tf32=tf32_status,
        throughput_qps=result.throughput_qps,
        total_host_walltime_s=result.total_host_walltime_s,
        metrics=metrics,
    )


def run_trtexec_evaluation(args: argparse.Namespace) -> ModelSummary:
    engine_path = Path(args.engine).expanduser().resolve()
    shapes = (
        args.shapes
        if args.shapes
        else f"{args.input_name}:{args.batch}x{args.ch}x{args.imgsz}x{args.imgsz}"
    )
    extra_args = shlex.split(args.trtexec_extra_args) if args.trtexec_extra_args else []
    trtexec_result = execute_trtexec(
        engine_path=engine_path,
        shapes=shapes,
        binary=args.trtexec,
        extra_args=extra_args,
        verbose=not args.no_trtexec_verbose,
    )

    if math.isnan(trtexec_result.throughput_qps):
        raise RuntimeError("Unable to parse throughput from trtexec output")

    metrics = normalise_metrics(trtexec_result.metrics)

    engine_dtype = trtexec_result.dtype
    if args.engine_dtype and args.engine_dtype.lower() != "auto":
        engine_dtype = args.engine_dtype

    tf32_status = trtexec_result.tf32
    if args.engine_tf32 and args.engine_tf32.lower() != "auto":
        tf32_status = parse_tf32_option(args.engine_tf32)

    return ModelSummary(
        title="TensorRT Engine",
        path=str(engine_path),
        dtype=engine_dtype,
        tf32=tf32_status,
        throughput_qps=trtexec_result.throughput_qps,
        total_host_walltime_s=trtexec_result.total_host_walltime_s,
        metrics=metrics,
    )


def execute_trtexec(
    engine_path: Path,
    shapes: str,
    binary: str,
    extra_args: Sequence[str],
    verbose: bool,
) -> TrtexecParseResult:
    command = [binary, f"--loadEngine={engine_path}"]
    if shapes:
        command.append(f"--shapes={shapes}")
    if verbose:
        command.append("--verbose")
    command.extend(extra_args)
    command_str = " ".join(shlex.quote(str(part)) for part in command)
    print(f"Running trtexec: {command_str}")

    try:
        completed = subprocess.run(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"trtexec binary '{binary}' was not found") from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or ""
        if output:
            sys.stderr.write(output)
        raise RuntimeError("trtexec command failed") from exc

    output_text = completed.stdout
    return parse_trtexec_output(output_text)


def parse_trtexec_output(output: str) -> TrtexecParseResult:
    throughput_qps = float("nan")
    total_host_walltime_s = float("nan")
    metrics: Dict[str, Dict[str, float]] = {}
    dtype_tokens: list[str] = []
    tf32_status: Optional[bool] = None

    for raw_line in output.splitlines():
        line = normalise_trtexec_line(raw_line)
        if not line:
            continue

        tf32_candidate = resolve_tf32_from_line(line)
        if tf32_candidate is not None:
            tf32_status = tf32_candidate

        format_match = re.search(
            r"(?:Input|Output)\(s\)\s*format\s*:\s*(.+)",
            line,
            re.IGNORECASE,
        )
        if format_match:
            dtype_tokens.extend(extract_dtype_tokens(format_match.group(1)))
            continue

        precision_match = re.search(r"Precision\s*:\s*(.+)", line, re.IGNORECASE)
        if precision_match:
            dtype_tokens.extend(extract_dtype_tokens(precision_match.group(1)))
            continue

        data_type_match = re.search(r"Data\s+type\s*:\s*(.+)", line, re.IGNORECASE)
        if data_type_match:
            dtype_tokens.extend(extract_dtype_tokens(data_type_match.group(1)))
            continue

        throughput_match = re.search(
            r"Throughput(?:\s*\(qps\))?\s*:\s*([\d.+\-eE]+)",
            line,
            re.IGNORECASE,
        )
        if throughput_match:
            throughput_qps = float(throughput_match.group(1))
            continue

        host_time_match = re.search(
            r"Total Host Walltime(?:\s*\((?P<unit>[a-z]+)\))?\s*:\s*(?P<value>[\d.+\-eE]+)",
            line,
            re.IGNORECASE,
        )
        if host_time_match:
            host_value = float(host_time_match.group("value"))
            host_unit = host_time_match.group("unit") or "s"
            total_host_walltime_s = convert_time(host_value, host_unit, target="s")
            continue

        if ":" not in line:
            continue

        label, rest = line.split(":", 1)
        label = label.strip()
        rest = rest.strip()
        if not label or not rest:
            continue

        metric_key = resolve_trtexec_label(label)
        if metric_key is None:
            continue

        stats = parse_latency_stats(rest)
        metrics[metric_key] = stats

    dtype_value = None
    if dtype_tokens:
        # Deduplicate while preserving order of appearance
        dtype_value = "/".join(dict.fromkeys(dtype_tokens))

    return TrtexecParseResult(
        throughput_qps=throughput_qps,
        total_host_walltime_s=total_host_walltime_s,
        metrics=metrics,
        dtype=dtype_value,
        tf32=tf32_status,
        raw_output=output,
    )


def normalise_trtexec_line(line: str) -> str:
    stripped = line.strip()
    while stripped.startswith("["):
        closing = stripped.find("]")
        if closing == -1:
            break
        stripped = stripped[closing + 1 :].lstrip()
    return stripped


def resolve_trtexec_label(label: str) -> Optional[str]:
    lowered = label.lower()
    for prefix, metric in LABEL_TO_METRIC:
        if lowered.startswith(prefix):
            return metric
    return None


def parse_latency_stats(payload: str) -> Dict[str, float]:
    stats = {name: float("nan") for name in STAT_NAMES}
    parts = [part.strip() for part in payload.split(",") if part.strip()]
    for part in parts:
        if "=" not in part:
            continue
        key, value_text = part.split("=", 1)
        key = key.strip().lower()
        value = extract_number(value_text)
        if value is None:
            continue
        unit = extract_unit(value_text)
        value_ms = convert_time(value, unit or "ms", target="ms")

        if key == "min":
            stats["min"] = value_ms
        elif key == "max":
            stats["max"] = value_ms
        elif key in {"mean", "avg", "average"}:
            stats["mean"] = value_ms
        elif key == "median":
            stats["median"] = value_ms
        elif key in {"p90", "p95", "p99"}:
            stats[key] = value_ms
        elif key.startswith("percentile"):
            percentile = extract_percentile(key)
            if percentile is None:
                continue
            if math.isclose(percentile, 90.0, rel_tol=1e-3, abs_tol=1e-3):
                stats["p90"] = value_ms
            elif math.isclose(percentile, 95.0, rel_tol=1e-3, abs_tol=1e-3):
                stats["p95"] = value_ms
            elif math.isclose(percentile, 99.0, rel_tol=1e-3, abs_tol=1e-3):
                stats["p99"] = value_ms
    return stats


def extract_number(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_unit(text: str) -> Optional[str]:
    match = re.search(r"(microseconds?|milliseconds?|ms|us|seconds?|sec|s)", text, re.IGNORECASE)
    if not match:
        return None
    unit = match.group(1).lower()
    if unit.startswith("micro"):
        return "us"
    if unit.startswith("milli"):
        return "ms"
    if unit.startswith("sec"):
        return "s"
    return unit


def extract_percentile(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_dtype_tokens(text: str) -> Sequence[str]:
    tokens = []
    # Split on common separators and strip TensorRT binding annotations (e.g. fp32:CHW)
    for chunk in re.split(r"[\s,;/]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        prefix = chunk.split(":", 1)[0].strip()
        if not prefix:
            continue
        if re.match(r"^[a-z0-9]+$", prefix, re.IGNORECASE):
            tokens.append(prefix.lower())
    return tokens


def resolve_tf32_from_line(line: str) -> Optional[bool]:
    lowered = line.lower()
    if "tf32" not in lowered:
        return None
    if any(keyword in lowered for keyword in ("disable", "disabled", "off", "not enable", "not support", "not avail")):
        return False
    if any(keyword in lowered for keyword in ("enable", "enabled", "on", "allow")):
        return True
    return None


def convert_time(value: float, unit: str, target: str) -> float:
    unit = (unit or "").lower()
    conversion_to_seconds = {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
    }
    seconds = value * conversion_to_seconds.get(unit, 1.0)
    if target == "s":
        return seconds
    if target == "ms":
        return seconds * 1000.0
    if target == "us":
        return seconds * 1_000_000.0
    return seconds


def parse_tf32_option(option: Optional[str]) -> Optional[bool]:
    if option is None:
        return None
    value = option.strip().lower()
    if value == "on":
        return True
    if value == "off":
        return False
    return None


def normalise_metrics(metrics: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    normalised: Dict[str, Dict[str, float]] = {}
    for metric in METRIC_NAMES:
        stats = {name: float("nan") for name in STAT_NAMES}
        if metrics and metric in metrics:
            for key, value in metrics[metric].items():
                if key in stats and value is not None:
                    stats[key] = float(value)
        normalised[metric] = stats
    return normalised


def format_dtype(dtype: Optional[str]) -> str:
    if dtype is None:
        return "Unknown"
    dtype_text = str(dtype).strip()
    return dtype_text or "Unknown"


def format_tf32(tf32: Optional[bool]) -> str:
    if tf32 is None:
        return "Unknown"
    return "On" if tf32 else "Off"


def format_table_value(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def format_csv_value(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6f}"


def format_csv_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value)


def format_ratio(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}x"


def safe_difference(a: float, b: float) -> float:
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(fa) or math.isnan(fb):
        return float("nan")
    return fa - fb


def safe_ratio(numerator: float, denominator: float) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(num) or math.isnan(den) or math.isclose(den, 0.0, rel_tol=1e-12, abs_tol=1e-12):
        return float("nan")
    return num / den


def print_summary(summary: ModelSummary) -> None:
    print(summary.title)
    print("-" * len(summary.title))
    print(f"Path: {summary.path}")
    print(f"Dtype: {format_dtype(summary.dtype)}")
    print(f"TF32: {format_tf32(summary.tf32)}")
    print()
    print("Performance summary")
    print("-------------------")
    print(f"Throughput (qps): {summary.throughput_qps:.4f}")
    if summary.total_host_walltime_s is None or math.isnan(summary.total_host_walltime_s):
        print("Total Host Walltime (s): N/A")
    else:
        print(f"Total Host Walltime (s): {summary.total_host_walltime_s:.6f}")
    print()
    print("Latency breakdown (ms)")
    print("----------------------")
    headers = ("Latency", "min", "max", "mean", "median", "p90", "p95", "p99")
    row_format = "{:<10} " + " ".join(["{:>10}"] * (len(headers) - 1))
    print(row_format.format(*headers))
    for metric in METRIC_NAMES:
        stats = summary.metrics.get(metric, {})
        row = [metric]
        for key in STAT_NAMES:
            row.append(format_table_value(stats.get(key, float("nan"))))
        print(row_format.format(*row))
    print()


def print_comparison(pt_summary: ModelSummary, engine_summary: ModelSummary) -> None:
    print("Comparison summary (engine - pt)")
    print("--------------------------------")
    throughput_delta = safe_difference(engine_summary.throughput_qps, pt_summary.throughput_qps)
    throughput_ratio = safe_ratio(engine_summary.throughput_qps, pt_summary.throughput_qps)
    print(f"Throughput delta (qps): {format_table_value(throughput_delta)}")
    print(f"Throughput ratio (engine/pt): {format_ratio(throughput_ratio)}")
    host_delta = safe_difference(
        engine_summary.total_host_walltime_s, pt_summary.total_host_walltime_s
    )
    print(f"Total Host Walltime delta (s): {format_table_value(host_delta)}")
    host_ratio = safe_ratio(
        engine_summary.total_host_walltime_s, pt_summary.total_host_walltime_s
    )
    print(f"Total Host Walltime ratio (engine/pt): {format_ratio(host_ratio)}")
    print()

    print("Latency delta (ms)")
    print("------------------")
    headers = ("Latency", "min", "max", "mean", "median", "p90", "p95", "p99")
    row_format = "{:<10} " + " ".join(["{:>10}"] * (len(headers) - 1))
    print(row_format.format(*headers))
    for metric in METRIC_NAMES:
        row = [metric]
        for key in STAT_NAMES:
            pt_value = pt_summary.metrics.get(metric, {}).get(key, float("nan"))
            engine_value = engine_summary.metrics.get(metric, {}).get(key, float("nan"))
            diff = safe_difference(engine_value, pt_value)
            row.append(format_table_value(diff))
        print(row_format.format(*row))
    print()

    print("Latency ratio (engine/pt)")
    print("-------------------------")
    print(row_format.format(*headers))
    for metric in METRIC_NAMES:
        row = [metric]
        for key in STAT_NAMES:
            pt_value = pt_summary.metrics.get(metric, {}).get(key, float("nan"))
            engine_value = engine_summary.metrics.get(metric, {}).get(key, float("nan"))
            ratio = safe_ratio(engine_value, pt_value)
            row.append(format_ratio(ratio))
        print(row_format.format(*row))
    print()


def write_compare_csv(
    pt_summary: ModelSummary,
    engine_summary: ModelSummary,
    output_dir: Path,
    output_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / output_name

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["metric", "pt", "engine", "engine_minus_pt", "engine_over_pt_ratio"]
        )
        writer.writerow(
            [
                "dtype",
                format_csv_text(pt_summary.dtype),
                format_csv_text(engine_summary.dtype),
                "",
                "",
            ]
        )
        writer.writerow(
            [
                "tf32",
                format_tf32(pt_summary.tf32),
                format_tf32(engine_summary.tf32),
                "",
                "",
            ]
        )
        writer.writerow(
            [
                "throughput_qps",
                format_csv_value(pt_summary.throughput_qps),
                format_csv_value(engine_summary.throughput_qps),
                format_csv_value(
                    safe_difference(
                        engine_summary.throughput_qps, pt_summary.throughput_qps
                    )
                ),
                format_csv_value(
                    safe_ratio(engine_summary.throughput_qps, pt_summary.throughput_qps)
                ),
            ]
        )
        writer.writerow(
            [
                "total_host_walltime_s",
                format_csv_value(pt_summary.total_host_walltime_s),
                format_csv_value(engine_summary.total_host_walltime_s),
                format_csv_value(
                    safe_difference(
                        engine_summary.total_host_walltime_s,
                        pt_summary.total_host_walltime_s,
                    )
                ),
                format_csv_value(
                    safe_ratio(
                        engine_summary.total_host_walltime_s,
                        pt_summary.total_host_walltime_s,
                    )
                ),
            ]
        )

        for metric in METRIC_NAMES:
            for key in STAT_NAMES:
                pt_value = pt_summary.metrics.get(metric, {}).get(key, float("nan"))
                engine_value = engine_summary.metrics.get(metric, {}).get(key, float("nan"))
                diff = safe_difference(engine_value, pt_value)
                writer.writerow(
                    [
                        f"{metric}_{key}",
                        format_csv_value(pt_value),
                        format_csv_value(engine_value),
                        format_csv_value(diff),
                        format_csv_value(safe_ratio(engine_value, pt_value)),
                    ]
                )

    return csv_path


def derive_output_name(pt_path: Path, engine_path: Path) -> str:
    pt_name = sanitise_filename(pt_path.stem or "pt_model")
    engine_name = sanitise_filename(engine_path.stem or "engine")
    return f"{pt_name}_vs_{engine_name}.csv"


def sanitise_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    cleaned = cleaned.strip("._")
    return cleaned or "model"


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    pt_summary = run_pt_evaluation(args)
    print_summary(pt_summary)

    engine_summary = run_trtexec_evaluation(args)
    print_summary(engine_summary)
    print_comparison(pt_summary, engine_summary)

    output_name = args.output_name or derive_output_name(Path(args.pt), Path(args.engine))
    csv_path = write_compare_csv(pt_summary, engine_summary, args.output_dir, output_name)
    print(f"Comparison CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
