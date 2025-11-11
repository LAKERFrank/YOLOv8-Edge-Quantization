#!/usr/bin/env python3
"""Aggregate trtexec artifacts into a concise report with visualisations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_latency_hist(latencies: pd.Series, out_png: Path) -> None:
    if latencies.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(latencies.values, bins=min(200, max(20, len(latencies) // 5)))
    plt.title("Latency distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Wrote {out_png}")


def plot_latency_percentiles(percentiles: List[Tuple[float, float]], out_png: Path) -> None:
    if not percentiles:
        return
    percentiles = sorted(percentiles, key=lambda x: x[0])
    xs, ys = zip(*percentiles)
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker="o")
    plt.title("Latency percentiles")
    plt.xlabel("Percentile")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Wrote {out_png}")


def build_report_text(summary: Dict, top_layers: Optional[pd.DataFrame]) -> str:
    lines = ["TensorRT Benchmark Summary", "===========================", ""]
    engine_info = summary.get("engine")
    if isinstance(engine_info, Mapping) and engine_info:
        path = engine_info.get("path")
        size_bytes = engine_info.get("size_bytes")
        size_mb = engine_info.get("size_megabytes")
        modified = engine_info.get("modified_iso")
        if path:
            lines.append(f"Engine path: {path}")
        if size_bytes is not None:
            try:
                size_bytes_int = int(size_bytes)
            except (TypeError, ValueError):
                size_bytes_int = None
            if size_bytes_int is not None:
                if size_mb is not None:
                    try:
                        size_mb_float = float(size_mb)
                    except (TypeError, ValueError):
                        size_mb_float = None
                    if size_mb_float is not None:
                        lines.append(
                            f"Engine size: {size_bytes_int} bytes ({size_mb_float:.2f} MiB)"
                        )
                    else:
                        lines.append(f"Engine size: {size_bytes_int} bytes")
                else:
                    lines.append(f"Engine size: {size_bytes_int} bytes")
        if modified:
            lines.append(f"Last modified: {modified}")
        lines.append("")
    run_config = summary.get("run_config")
    if isinstance(run_config, Mapping) and run_config:
        lines.append("Run configuration:")
        config_parts = []
        for label, key in (
            ("batch", "batch"),
            ("iterations", "iterations"),
            ("warmup", "warmup"),
            ("avg_runs", "avg_runs"),
        ):
            value = run_config.get(key)
            if value is None:
                continue
            config_parts.append(f"{label}={value}")
        use_cuda = run_config.get("use_cuda_graph")
        if use_cuda is not None:
            config_parts.append(f"use_cuda_graph={'yes' if use_cuda else 'no'}")
        if config_parts:
            lines.append("  " + ", ".join(config_parts))
        timestamp = run_config.get("timestamp_iso")
        if timestamp:
            lines.append(f"  Timestamp: {timestamp}")
        trtexec_bin = run_config.get("trtexec_binary")
        if trtexec_bin:
            lines.append(f"  trtexec binary: {trtexec_bin}")
        command = run_config.get("command")
        if command:
            lines.append(f"  trtexec command: {command}")
        lines.append("")
    throughput = summary.get("throughput_qps")
    if throughput is not None:
        lines.append(f"Throughput: {throughput:.3f} samples/sec")
    latency = summary.get("latency", {})
    if latency:
        lines.append(
            "Latency (ms): min={min:.3f}, mean={mean:.3f}, median={median:.3f}, p95={p95:.3f}, p99={p99:.3f}".format(
                min=latency.get("min_ms", 0.0),
                mean=latency.get("mean_ms", 0.0),
                median=latency.get("median_ms", 0.0),
                p95=latency.get("percentiles_ms", {}).get("p95", 0.0),
                p99=latency.get("percentiles_ms", {}).get("p99", 0.0),
            )
        )
    if summary.get("enqueue"):
        enqueue = summary["enqueue"]
        lines.append(
            "Enqueue mean: {mean:.3f} ms, std={std:.3f} ms".format(
                mean=enqueue.get("mean_ms", 0.0),
                std=enqueue.get("std_ms", 0.0),
            )
        )
    if summary.get("compute"):
        compute = summary["compute"]
        lines.append(
            "Compute mean: {mean:.3f} ms, std={std:.3f} ms".format(
                mean=compute.get("mean_ms", 0.0),
                std=compute.get("std_ms", 0.0),
            )
        )
    if top_layers is not None and not top_layers.empty:
        lines.append("")
        lines.append("Top layers (by avg time):")
        for _, row in top_layers.head(10).iterrows():
            lines.append(
                f"- {row['name']}: {row['avg_time_ms']:.4f} ms ({row.get('percent_of_total', 0.0):.2f}% of total)"
            )
    return "\n".join(lines)


def parse_percentiles(summary: Dict) -> List[Tuple[float, float]]:
    percentiles_dict = summary.get("latency", {}).get("percentiles_ms", {})
    parsed: List[Tuple[float, float]] = []
    for key, value in percentiles_dict.items():
        if not key.startswith("p"):
            continue
        try:
            pct = float(key[1:].replace("_", "."))
        except ValueError:
            continue
        parsed.append((pct, float(value)))
    return parsed
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consolidated benchmark report")
    parser.add_argument("--artifacts", required=True, type=Path, help="Directory with trtexec outputs")
    args = parser.parse_args()

    artifacts_dir = args.artifacts
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts directory {artifacts_dir} does not exist")

    summary_path = artifacts_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit("summary.json not found. Run parse_trtexec_times.py first.")
    summary = load_json(summary_path)

    lat_series_path = artifacts_dir / "latency_series.csv"
    latencies = None
    if lat_series_path.exists():
        latencies = pd.read_csv(lat_series_path).squeeze()
        if not isinstance(latencies, pd.Series):
            latencies = pd.Series(latencies)
        latencies = latencies.dropna()
        plot_latency_hist(latencies, artifacts_dir / "latency_hist.png")
    else:
        print("latency_series.csv not found; skipping latency histogram.")

    percentiles = parse_percentiles(summary)
    if percentiles:
        plot_latency_percentiles(percentiles, artifacts_dir / "percentiles.png")
    else:
        print("No percentile information available to plot.")

    per_layer_csv = artifacts_dir / "per_layer_times.csv"
    per_layer_df = None
    if per_layer_csv.exists():
        per_layer_df = pd.read_csv(per_layer_csv)
        per_layer_df = per_layer_df.sort_values("avg_time_ms", ascending=False)
    else:
        print("per_layer_times.csv not found; skipping per-layer summary.")

    compare_path = artifacts_dir / "output_diff.json"
    compare_summary = load_json(compare_path) if compare_path.exists() else None

    report_data = {
        "summary": summary,
        "top_layers": per_layer_df.head(10).to_dict(orient="records") if per_layer_df is not None else [],
    }
    if compare_summary:
        report_data["output_comparison"] = compare_summary

    report_json = artifacts_dir / "report.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Wrote {report_json}")

    report_text = build_report_text(summary, per_layer_df)
    report_md = artifacts_dir / "report.md"
    report_md.write_text(report_text + "\n", encoding="utf-8")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
