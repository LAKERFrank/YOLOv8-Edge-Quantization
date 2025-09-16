#!/usr/bin/env python3
"""Parse trtexec exportTimes JSON and derive latency / throughput statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSequence, Sequence

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


def summarise(parsed: Mapping[str, Sequence[float]], batch: int) -> Dict[str, object]:
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
    args = parser.parse_args()

    times_path: Path = args.times_json
    outdir = args.outdir or times_path.parent

    parsed = parse_times_json(times_path)
    summary = summarise(parsed, args.batch)

    write_outputs(parsed, summary, outdir)
    print(f"Wrote summary to {outdir}")


if __name__ == "__main__":
    main()
