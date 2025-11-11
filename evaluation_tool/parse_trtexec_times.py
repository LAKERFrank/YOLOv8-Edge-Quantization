#!/usr/bin/env python3
"""Parse trtexec exportTimes JSON and derive latency / throughput statistics."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence

import numpy as np
import pandas as pd

LAT_KEYS = [
    "latency",
    "latency_ms",
    "latencyMs",
    "latencyAverage",
    "latencyAverageMs",
    "iterationMs",
    "time",
    "time_ms",
    "timeMs",
    "gpu_time",
    "gpu_time_ms",
    "gpuTimeMs",
    "duration",
    "duration_ms",
    "durationMs",
]

ENQUEUE_KEYS = [
    "enqueue",
    "enqueue_ms",
    "enqueueMs",
    "host_time",
    "host_time_ms",
    "hostTimeMs",
]

COMPUTE_KEYS = [
    "compute",
    "compute_ms",
    "computeMs",
    "gpu",
    "gpu_ms",
    "gpuMs",
    "device_time",
    "device_time_ms",
]

END_TO_END_KEYS = [
    "endToEnd",
    "end_to_end",
    "total",
    "overall",
    "overall_ms",
    "overallMs",
]

PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 97, 99, 99.5, 99.9, 100]


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _normalise_engine_metadata(raw: Mapping[str, object]) -> Dict[str, object]:
    if not isinstance(raw, Mapping):
        return {}

    def _first(*keys: str):
        for key in keys:
            if key in raw and raw[key] not in (None, ""):
                return raw[key]
        return None

    result: Dict[str, object] = {}

    path_value = _first("path", "engine_path")
    path_str: Optional[str] = None
    if path_value is not None:
        path_obj = Path(str(path_value)).expanduser()
        try:
            path_str = str(path_obj.resolve(strict=False))
        except OSError:
            path_str = str(path_obj)
        result["path"] = path_str

    filename_value = _first("filename", "engine_filename")
    if filename_value is None and path_str:
        filename_value = Path(path_str).name
    if filename_value is not None:
        result["filename"] = str(filename_value)

    size_bytes = _safe_int(_first("size_bytes", "engine_size_bytes"))
    if size_bytes is not None and size_bytes >= 0:
        result["size_bytes"] = size_bytes

    size_megabytes = _safe_float(
        _first("size_megabytes", "engine_size_megabytes", "size_mib", "size_mebibytes")
    )
    if size_megabytes is None and size_bytes is not None:
        size_megabytes = size_bytes / (1024 ** 2)
    if size_megabytes is not None:
        result["size_megabytes"] = float(size_megabytes)

    modified_value = _first("modified_iso", "engine_modified_iso", "modified", "mtime_iso")
    if modified_value is not None:
        result["modified_iso"] = str(modified_value)
    elif path_str:
        try:
            stat_result = Path(path_str).stat()
        except OSError:
            pass
        else:
            result["modified_iso"] = dt.datetime.fromtimestamp(stat_result.st_mtime).astimezone().isoformat()

    return result


def _normalise_run_config(raw: Mapping[str, object]) -> Dict[str, object]:
    if not isinstance(raw, Mapping):
        return {}

    def _first(*keys: str):
        for key in keys:
            if key in raw and raw[key] not in (None, ""):
                return raw[key]
        return None

    result: Dict[str, object] = {}

    timestamp = _first("timestamp_iso", "timestamp", "run_timestamp")
    if timestamp is not None:
        result["timestamp_iso"] = str(timestamp)

    for field, aliases in (
        ("batch", ("batch", "batch_size", "run_batch")),
        ("iterations", ("iterations", "iters", "benchmark_iterations")),
        ("warmup", ("warmup", "warmUp", "warmup_iterations")),
        ("avg_runs", ("avg_runs", "avgRuns", "average_runs")),
    ):
        value = _safe_int(_first(*aliases))
        if value is not None:
            result[field] = value

    use_cuda_graph = _safe_bool(_first("use_cuda_graph", "useCudaGraph", "cuda_graph"))
    if use_cuda_graph is not None:
        result["use_cuda_graph"] = use_cuda_graph

    for field, aliases in (
        ("trtexec_binary", ("trtexec_binary", "trtexec", "binary")),
        ("command", ("command", "cmd", "trtexec_command")),
        ("outdir", ("outdir", "output_dir")),
    ):
        value = _first(*aliases)
        if value is not None:
            result[field] = str(value)

    return result


def gather_engine_info(
    engine_path: Optional[Path], metadata_candidates: Sequence[Path]
) -> Optional[Dict[str, object]]:
    if engine_path is not None:
        candidate = engine_path.expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            print(f"[parse_trtexec_times] Engine file '{engine_path}' not found", file=sys.stderr)
        else:
            stat_result = resolved.stat()
            info = {
                "path": str(resolved),
                "filename": resolved.name,
                "size_bytes": int(stat_result.st_size),
                "size_megabytes": stat_result.st_size / (1024 ** 2),
                "modified_iso": dt.datetime.fromtimestamp(stat_result.st_mtime).astimezone().isoformat(),
            }
            return info

    seen: set[str] = set()
    for meta_path in metadata_candidates:
        if meta_path is None:
            continue
        candidate = meta_path.expanduser()
        try:
            candidate_str = str(candidate.resolve(strict=False))
        except OSError:
            candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        candidate_path = Path(candidate_str)
        if not candidate_path.exists():
            continue
        try:
            with candidate_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[parse_trtexec_times] Failed to read engine metadata from {candidate_path}: {exc}",
                file=sys.stderr,
            )
            continue
        info = _normalise_engine_metadata(raw)
        if info:
            return info

    return None


def gather_run_config(candidates: Sequence[Path]) -> Optional[Dict[str, object]]:
    seen: set[str] = set()
    for path in candidates:
        if path is None:
            continue
        candidate = path.expanduser()
        try:
            candidate_str = str(candidate.resolve(strict=False))
        except OSError:
            candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        candidate_path = Path(candidate_str)
        if not candidate_path.exists():
            continue
        try:
            with candidate_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[parse_trtexec_times] Failed to read run configuration from {candidate_path}: {exc}",
                file=sys.stderr,
            )
            continue
        config = _normalise_run_config(raw)
        if config:
            return config
    return None


def _coerce_to_float(value) -> float | None:
    """Best-effort conversion of values from trtexec JSON into milliseconds."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        unit_multiplier = 1.0
        for suffix, multiplier in ("ms", 1.0), ("microseconds", 1e-3), ("us", 1e-3), ("s", 1000.0):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                unit_multiplier = multiplier
                break
        try:
            return float(cleaned) * unit_multiplier
        except ValueError:
            return None
    if isinstance(value, Mapping):
        for key in ("avg", "average", "mean", "time", "ms", "value"):
            if key in value:
                nested = _coerce_to_float(value[key])
                if nested is not None:
                    return nested
    return None


def _append_from_entry(
    entry: Mapping,
    keys: Sequence[str],
    target: MutableSequence[float],
) -> bool:
    for key in keys:
        if key in entry:
            parsed = _coerce_to_float(entry[key])
            if parsed is not None:
                target.append(parsed)
                return True
    return False


def _parse_list_of_entries(
    items: Iterable,
    latencies: MutableSequence[float],
    enqueue: MutableSequence[float],
    compute: MutableSequence[float],
    end_to_end: MutableSequence[float],
) -> None:
    for element in items:
        if isinstance(element, Mapping):
            _append_from_entry(element, LAT_KEYS, latencies)
            _append_from_entry(element, ENQUEUE_KEYS, enqueue)
            _append_from_entry(element, COMPUTE_KEYS, compute)
            _append_from_entry(element, END_TO_END_KEYS, end_to_end)
        else:
            parsed = _coerce_to_float(element)
            if parsed is not None:
                latencies.append(parsed)


def _parse_per_iteration_dict(
    node: Mapping,
    latencies: MutableSequence[float],
    enqueue: MutableSequence[float],
    compute: MutableSequence[float],
    end_to_end: MutableSequence[float],
) -> bool:
    found = False
    for keys, target in (
        (LAT_KEYS, latencies),
        (ENQUEUE_KEYS, enqueue),
        (COMPUTE_KEYS, compute),
        (END_TO_END_KEYS, end_to_end),
    ):
        for key in keys:
            if key in node:
                value = node[key]
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                    continue
                series = [_coerce_to_float(v) for v in value]
                filtered = [v for v in series if v is not None]
                if filtered:
                    target.extend(filtered)
                    found = True
                    break
    return found


def maybe_convert_seconds(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    if np.isnan(arr).any():
        arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return []
    mean_val = float(np.mean(arr))
    if mean_val < 0.1:  # assume seconds -> convert to milliseconds
        arr *= 1000.0
    return arr.tolist()


def compute_stats(series: Sequence[float]) -> Dict[str, float]:
    if not series:
        return {}
    arr = np.asarray(series, dtype=np.float64)
    stats: Dict[str, float] = {
        "count": int(arr.size),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=0)),
        "median_ms": float(np.median(arr)),
    }
    percentile_values = np.percentile(arr, PERCENTILES)
    stats["percentiles_ms"] = {
        f"p{str(p).replace('.', '_')}": float(v)
        for p, v in zip(PERCENTILES, percentile_values)
    }
    return stats


def parse_times_json(path: Path) -> Dict[str, List[float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    latencies: List[float] = []
    enqueue: List[float] = []
    compute: List[float] = []
    end_to_end: List[float] = []

    if isinstance(data, Mapping):
        extracted = False
        for key in ("times", "iterations", "runs"):
            if key in data and isinstance(data[key], Sequence):
                _parse_list_of_entries(data[key], latencies, enqueue, compute, end_to_end)
                extracted = True
        for key in ("perIteration", "per_iteration", "iteration_times"):
            if key in data and isinstance(data[key], Mapping):
                found = _parse_per_iteration_dict(data[key], latencies, enqueue, compute, end_to_end)
                extracted = extracted or found
        if not extracted:
            # walk nested structures as a fallback
            stack = [data]
            while stack:
                node = stack.pop()
                if isinstance(node, Mapping):
                    if any(k in node for k in LAT_KEYS):
                        _append_from_entry(node, LAT_KEYS, latencies)
                        _append_from_entry(node, ENQUEUE_KEYS, enqueue)
                        _append_from_entry(node, COMPUTE_KEYS, compute)
                        _append_from_entry(node, END_TO_END_KEYS, end_to_end)
                    for value in node.values():
                        if isinstance(value, (Mapping, Sequence)):
                            stack.append(value)
                elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
                    _parse_list_of_entries(node, latencies, enqueue, compute, end_to_end)
    elif isinstance(data, Sequence):
        _parse_list_of_entries(data, latencies, enqueue, compute, end_to_end)

    latencies = maybe_convert_seconds(latencies)
    enqueue = maybe_convert_seconds(enqueue)
    compute = maybe_convert_seconds(compute)
    end_to_end = maybe_convert_seconds(end_to_end)

    return {
        "latencies_ms": latencies,
        "enqueue_ms": enqueue,
        "compute_ms": compute,
        "end_to_end_ms": end_to_end,
    }


def summarise(
    parsed: Mapping[str, Sequence[float]],
    batch: int,
    engine_info: Optional[Mapping[str, object]] = None,
    run_config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    latency_series = parsed.get("latencies_ms", [])
    latency_stats = compute_stats(latency_series)
    throughput = None
    if latency_stats:
        mean_ms = latency_stats["mean_ms"]
        if mean_ms > 0:
            throughput = batch * (1000.0 / mean_ms)
    summary: Dict[str, object] = {
        "batch": batch,
        "samples": int(latency_stats.get("count", 0)),
        "throughput_qps": throughput,
        "latency": latency_stats,
        "enqueue": compute_stats(parsed.get("enqueue_ms", [])),
        "compute": compute_stats(parsed.get("compute_ms", [])),
        "end_to_end": compute_stats(parsed.get("end_to_end_ms", [])),
    }
    if engine_info:
        summary["engine"] = dict(engine_info)
    if run_config:
        summary["run_config"] = dict(run_config)
    return summary


def write_outputs(
    parsed: Mapping[str, Sequence[float]],
    summary: Mapping[str, object],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    df = pd.json_normalize(summary)
    df.to_csv(outdir / "summary.csv", index=False)

    engine_info = summary.get("engine")
    if isinstance(engine_info, Mapping):
        metadata_path = outdir / "engine_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(engine_info, f, indent=2)

    run_config = summary.get("run_config")
    if isinstance(run_config, Mapping):
        run_config_path = outdir / "run_config.json"
        with run_config_path.open("w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

    latencies = parsed.get("latencies_ms", [])
    if latencies:
        series = pd.Series(latencies, name="latency_ms")
        series.to_csv(outdir / "latency_series.csv", index=False)
        percentile_df = pd.DataFrame(
            {
                "percentile": PERCENTILES,
                "latency_ms": [
                    summary["latency"]["percentiles_ms"][f"p{str(p).replace('.', '_')}"]
                    for p in PERCENTILES
                ],
            }
        )
        percentile_df.to_csv(outdir / "latency_percentiles.csv", index=False)

    enqueue = parsed.get("enqueue_ms", [])
    if enqueue:
        pd.Series(enqueue, name="enqueue_ms").to_csv(outdir / "enqueue_series.csv", index=False)
    compute = parsed.get("compute_ms", [])
    if compute:
        pd.Series(compute, name="compute_ms").to_csv(outdir / "compute_series.csv", index=False)
    end_to_end = parsed.get("end_to_end_ms", [])
    if end_to_end:
        pd.Series(end_to_end, name="end_to_end_ms").to_csv(outdir / "end_to_end_series.csv", index=False)



def main() -> None:
    parser = argparse.ArgumentParser(description="Parse trtexec exportTimes output.")
    parser.add_argument("times_json", type=Path, help="Path to trtexec --exportTimes JSON")
    parser.add_argument("batch", type=int, help="Batch size used during benchmarking")
    parser.add_argument(
        "outdir",
        nargs="?",
        type=Path,
        default=None,
        help="Output directory (defaults to JSON parent)",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=None,
        help="TensorRT engine file to record size information",
    )
    parser.add_argument(
        "--engine-metadata",
        dest="engine_metadata",
        type=Path,
        default=None,
        help="Path to engine metadata JSON (defaults to <outdir>/engine_metadata.json)",
    )
    parser.add_argument(
        "--run-config",
        dest="run_config",
        type=Path,
        default=None,
        help="Path to run_config.json (defaults to <outdir>/run_config.json)",
    )
    args = parser.parse_args()

    times_path: Path = args.times_json
    outdir = args.outdir or times_path.parent

    parsed = parse_times_json(times_path)
    metadata_candidates: List[Path] = []
    if args.engine_metadata is not None:
        metadata_candidates.append(args.engine_metadata)
    metadata_candidates.append(outdir / "engine_metadata.json")
    parent_candidate = times_path.parent / "engine_metadata.json"
    if parent_candidate not in metadata_candidates:
        metadata_candidates.append(parent_candidate)

    engine_info = gather_engine_info(args.engine, metadata_candidates)

    run_config_candidates: List[Path] = []
    if args.run_config is not None:
        run_config_candidates.append(args.run_config)
    run_config_candidates.append(outdir / "run_config.json")
    run_parent_candidate = times_path.parent / "run_config.json"
    if run_parent_candidate not in run_config_candidates:
        run_config_candidates.append(run_parent_candidate)

    run_config = gather_run_config(run_config_candidates)
    if run_config and "batch" in run_config:
        cfg_batch = _safe_int(run_config.get("batch"))
        if cfg_batch not in (None, args.batch):
            print(
                f"[parse_trtexec_times] Warning: run_config batch {cfg_batch} differs from provided batch {args.batch}. Using CLI value.",
                file=sys.stderr,
            )

    summary = summarise(parsed, args.batch, engine_info, run_config)

    write_outputs(parsed, summary, outdir)
    print(f"Wrote summary to {outdir}")


if __name__ == "__main__":
    main()
