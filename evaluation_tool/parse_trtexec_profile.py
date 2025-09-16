#!/usr/bin/env python3
"""Parse trtexec exportProfile output and build per-layer timing tables/plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Mapping, MutableSet, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIME_KEYS = [
    "time_ms",
    "timeMs",
    "average_time_ms",
    "avg_time_ms",
    "avgTimeMs",
    "average_time",
    "avg",  # fallback
    "time",
    "execution_time_ms",
    "selfTimeMs",
]

MIN_KEYS = ["min_time_ms", "minTimeMs", "min", "min_ms"]
MAX_KEYS = ["max_time_ms", "maxTimeMs", "max", "max_ms"]
COUNT_KEYS = ["count", "calls", "invocations"]


def _coerce_to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        multiplier = 1.0
        for suffix, factor in ("ms", 1.0), ("microseconds", 1e-3), ("us", 1e-3), ("s", 1000.0):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                multiplier = factor
                break
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None
    if isinstance(value, Mapping):
        for key in ("avg", "average", "mean", "time", "ms", "value"):
            if key in value:
                nested = _coerce_to_float(value[key])
                if nested is not None:
                    return nested
    return None


def _extract_first(entry: Mapping, keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in entry:
            parsed = _coerce_to_float(entry[key])
            if parsed is not None:
                return parsed
    return None


def _gather_entries(node, seen: MutableSet[str], entries: List[dict]) -> None:
    if isinstance(node, Mapping):
        name = node.get("name") or node.get("layer") or node.get("id")
        if name is not None:
            avg = _extract_first(node, TIME_KEYS)
            if avg is not None:
                min_v = _extract_first(node, MIN_KEYS)
                max_v = _extract_first(node, MAX_KEYS)
                count = _extract_first(node, COUNT_KEYS)
                key = f"{name}|{avg}"  # deduplicate duplicates by avg time
                if key not in seen:
                    seen.add(key)
                    entries.append(
                        {
                            "name": str(name),
                            "avg_time_ms": float(avg),
                            "min_time_ms": float(min_v) if min_v is not None else None,
                            "max_time_ms": float(max_v) if max_v is not None else None,
                            "calls": int(count) if count is not None else None,
                        }
                    )
        for value in node.values():
            if isinstance(value, (Mapping, Sequence)) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                _gather_entries(value, seen, entries)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for item in node:
            _gather_entries(item, seen, entries)



def parse_profile(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries: List[dict] = []
    _gather_entries(data, set(), entries)
    if not entries:
        return pd.DataFrame(columns=["name", "avg_time_ms", "min_time_ms", "max_time_ms", "calls"])

    df = pd.DataFrame(entries)
    df = df.dropna(subset=["avg_time_ms"])
    df["avg_time_ms"] = pd.to_numeric(df["avg_time_ms"], errors="coerce")
    df["min_time_ms"] = pd.to_numeric(df.get("min_time_ms"), errors="coerce")
    df["max_time_ms"] = pd.to_numeric(df.get("max_time_ms"), errors="coerce")
    df["calls"] = pd.to_numeric(df.get("calls"), errors="coerce")
    df = df.dropna(subset=["avg_time_ms"])
    df = df.sort_values("avg_time_ms", ascending=False)
    total_time = df["avg_time_ms"].sum()
    if total_time > 0:
        df["percent_of_total"] = df["avg_time_ms"] / total_time * 100.0
    else:
        df["percent_of_total"] = 0.0
    return df


def plot_topk(df: pd.DataFrame, out_png: Path, topk: int) -> None:
    top = df.head(topk)
    if top.empty:
        return
    plt.figure(figsize=(12, max(4, top.shape[0] * 0.35)))
    y_labels = top["name"].astype(str)[::-1]
    times = top["avg_time_ms"][::-1]
    plt.barh(y_labels, times)
    plt.xlabel("Average time (ms)")
    plt.ylabel("Layer")
    plt.title(f"Top {top.shape[0]} layers by average time")
    for idx, (layer, time_ms, percent) in enumerate(
        zip(y_labels, times, top["percent_of_total"][::-1])
    ):
        plt.text(time_ms, idx, f" {time_ms:.3f} ms ({percent:.2f}%)", va="center")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse trtexec profile output.")
    parser.add_argument("profile_json", type=Path, help="Path to trtexec --exportProfile JSON")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (defaults to JSON parent)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Number of slowest layers to plot (default: 30)",
    )
    args = parser.parse_args()

    profile_path = args.profile_json
    outdir = args.outdir or profile_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = parse_profile(profile_path)
    if df.empty:
        raise SystemExit("No per-layer timing information found in profile.json")

    csv_path = outdir / "per_layer_times.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} with {len(df)} layers")

    plot_topk(df, outdir / "per_layer_time.png", args.topk)


if __name__ == "__main__":
    main()
