# TensorRT `trtexec` Evaluation Toolkit

This folder contains a small toolkit for running reproducible TensorRT engine benchmarks using `trtexec`, parsing the generated artefacts, comparing engine outputs, and building presentation-ready charts/reports.

## Directory layout

- `run_trtexec.sh` &ndash; Bash wrapper that standardises `trtexec` runs and stores JSON artefacts inside an output folder.
- `parse_trtexec_times.py` &ndash; Parses `times.json` produced via `--exportTimes`, computes latency / throughput statistics, and exports summary CSV/JSON files.
- `parse_trtexec_profile.py` &ndash; Parses per-layer timing information from `profile.json` and produces CSV + bar-chart visualisations.
- `compare_model_outputs.py` &ndash; Benchmarks a YOLOv8 Pose PyTorch model and a TensorRT engine, summarises throughput/latency, and exports comparison CSVs.
- `generate_report.py` &ndash; Aggregates the artefacts into a consolidated report (`report.json`/`report.md`) and generates latency/per-layer plots.

> **Note:** Python scripts expect Python ≥ 3.8. Optional features require additional dependencies listed below.

## Prerequisites

1. Linux environment with TensorRT installed and `trtexec` accessible via `$PATH`.
2. Python ≥ 3.8 with the following packages:
   - Required: `numpy`, `pandas`, `matplotlib`.
   - Optional: `scipy`, `tqdm` for extended analyses (not required by default scripts).
   - `compare_model_outputs.py` additionally requires `torch` for the PyTorch benchmark and relies on the `trtexec` CLI for engine timing (TensorRT Python bindings are not required).
3. (Recommended) Create a Python virtual environment and install the dependencies, e.g.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas matplotlib torch
   ```

## Typical workflow

1. **Run `trtexec` and capture artefacts**
   ```bash
   ./evaluation_tool/run_trtexec.sh \
     --engine path/to/model.engine \
     --batch 1 \
     --iters 300 \
     --warmup 500 \
     --useCudaGraph \
     --outdir artifacts/my_model
   ```
  The script saves `times.json`, `profile.json` (unless profiling is disabled), `engine_metadata.json`, `run_config.json`, and a `trtexec_stdout.log` file into the specified `--outdir` (created automatically). `engine_metadata.json` records the absolute engine path, file size, and last modified timestamp for downstream reports, while `run_config.json` captures the run settings (batch size, iterations, warmup, CUDA Graph usage, profiling flags, and the resolved `trtexec` command).

  By default the wrapper requests per-layer profiling via `--dumpProfile` **and** automatically appends `--separateProfileRun` so that TensorRT collects profile data in a dedicated pass without suppressing the end-to-end timing statistics. Disable profiling entirely with `--disableProfile`, or keep the profiler but skip the extra pass with `--disableSeparateProfileRun` if you explicitly need the legacy behaviour.

  > **Note:** TensorRT executes the shapes embedded inside a serialized engine, so the batch size provided to the wrapper is stored for reporting but not forwarded to `trtexec`. If your workflow requires explicitly passing `--batch=<N>` to `trtexec`, forward it via `--extra "--batch=<N>"` or arguments after `--`.

2. **Parse latency statistics**
   ```bash
   python evaluation_tool/parse_trtexec_times.py artifacts/my_model/times.json 1
   ```
   Outputs:
  - `summary.json` / `summary.csv` containing throughput and latency statistics (min/mean/median/percentiles) plus engine file metadata (path, size in bytes/MB, last modified time).
   - `latency_series.csv`, `latency_percentiles.csv`, and optional enqueue/compute time series if available.

3. **Analyse per-layer timings**
   ```bash
   python evaluation_tool/parse_trtexec_profile.py artifacts/my_model/profile.json
   ```
   Outputs:
   - `per_layer_times.csv` sorted by average time.
   - `per_layer_time.png` visualising the top-k (default 30) slowest layers.

4. **Generate consolidated report**
   ```bash
   python evaluation_tool/generate_report.py --artifacts artifacts/my_model
   ```
   Outputs:
   - `latency_hist.png`, `percentiles.png`, and (if available) `per_layer_time.png`.
   - `report.json` aggregating summary metrics, top layers, and optional output comparison results.
   - `report.md` human-readable summary.

# Compare FP32 vs INT8 outputs

Use `evaluation_tool/compare_model_outputs.py` to benchmark a YOLOv8 Pose PyTorch checkpoint against a TensorRT engine and inspect the performance delta between FP32 and INT8 deployments.

## Quick start

```bash
python evaluation_tool/compare_model_outputs.py \
  --pt weights/pose_fp32.pt \
  --engine trt_quant/engine/pose_int8_minmax.engine \
  --imgsz 640 \
  --batch 1 \
  --ch 1 \
  --shapes images:1x1x640x640 \
  --pt-iters 2000 \
  --pt-warmup 200
```

- The PyTorch measurements reuse `bench_pt_yolo_pose.py`. Tune `--pt-dtype` (`fp32`, `fp16`), `--pt-device`, `--pt-iters`, `--pt-warmup`, and `--pt-no-tf32` to mirror your deployment settings. Add `--pt-ultra` to load checkpoints via `ultralytics.YOLO` when required.
- TensorRT statistics are collected by launching `trtexec`. Adjust `--input-name` and `--shapes` to match the engine bindings, append extra CLI flags with `--trtexec-extra-args`, and disable verbose logging with `--no-trtexec-verbose` if you prefer compact output.
- The script attempts to infer the engine precision and TF32 state from the `trtexec` logs. Override them explicitly via `--engine-dtype` or `--engine-tf32` when necessary.

## Output artefacts

- Two per-model summaries (PyTorch and TensorRT) are printed to the console, each listing the resolved dtype, TF32 state, throughput, total host walltime, and a latency table (min/max/mean/median/percentiles for host, H2D, GPU, and D2H).
- A comparison section follows, highlighting throughput and host walltime deltas plus latency differences/ratios (`engine - pt` and `engine / pt`).
- All metrics are exported to `artifacts/compare/<pt_stem>_vs_<engine_stem>.csv` by default. Customise the destination with `--output-dir`/`--output-name`. The CSV includes the original PyTorch/TensorRT numbers alongside `engine_minus_pt` and `engine_over_pt_ratio` columns for every statistic.
- Reuse the generated CSVs for downstream visualisations or spreadsheets; each row is labelled with the corresponding latency component and percentile.

## Tips

- Ensure the batch size, image size, and channel count are identical for both benchmarks to keep comparisons fair.
- Lock GPU clocks (e.g. on Jetson devices) during measurements to reduce variance across runs.
- When benchmarking multiple engines, store them in distinct CSVs under `artifacts/compare/` to simplify longitudinal tracking.

## Script details & options

### `run_trtexec.sh`
- Accepts `--shape <tensor:dimx...>` to forward dynamic shape definitions to `trtexec`.
- Adds `--dumpProfile`, `--exportProfile`, and (by default) `--separateProfileRun` so `times.json` remains available even when profiling is enabled. Use `--disableProfile` to skip per-layer artefacts, or `--disableSeparateProfileRun` if you want to profile within the primary timing pass.
- Use `--extra "--memPoolSize=workspace:4096"` or append arguments after `--` to pass additional flags directly.
- Logs the exact command and all `trtexec` output to `<outdir>/trtexec_stdout.log`. On failures it
  exits with the original `trtexec` status code and prints the last 20 log lines to stderr for
  quicker debugging.
- Emits `<outdir>/engine_metadata.json` summarising the engine path, size, and modification timestamp, plus `<outdir>/run_config.json` documenting the run parameters and resolved `trtexec` command. These metadata files are automatically consumed by the parsing/report scripts.

### `parse_trtexec_times.py`
- Usage: `python parse_trtexec_times.py <times.json> <batch> [outdir]`.
- Robust to minor schema changes (`times`, `perIteration`, nested dicts) and automatically converts units to milliseconds.
- Produces `latency_percentiles.csv` for plotting percentile curves.
- Automatically loads `<outdir>/engine_metadata.json` (or accepts `--engine path/to/model.engine`) to capture file size information in `summary.json` / `summary.csv`.
- Reads `<outdir>/run_config.json` (override with `--run-config <path>`) to embed the recorded benchmark settings into the summary and warn if the stored batch size differs from the command-line argument.

### `parse_trtexec_profile.py`
- Usage: `python parse_trtexec_profile.py <profile.json> [--outdir DIR] [--topk N]`.
- Handles multiple nesting layouts (`layers`, `profile`, generic nested dicts) and deduplicates repeated entries by `(name, avg_time)` pairs.

### `compare_model_outputs.py`
- Usage: `python compare_model_outputs.py --pt <model.pt> --engine <model.engine> [--imgsz 640] [--batch 1] [--ch 1]`.
- PyTorch benchmarking flags mirror `bench_pt_yolo_pose.py`: adjust `--pt-iters`, `--pt-warmup`, `--pt-device`, `--pt-dtype`, `--pt-no-tf32`, and `--pt-ultra` as needed.
- TensorRT benchmarking flags: `--input-name`, `--shapes`, `--trtexec`, `--trtexec-extra-args`, `--no-trtexec-verbose`, `--engine-dtype`, and `--engine-tf32`.
- Output controls: `--output-dir` and `--output-name` determine where comparison CSVs are stored (default `artifacts/compare/`).
- CSVs capture absolute values, `engine_minus_pt`, and `engine_over_pt_ratio` columns for every reported metric alongside the console summaries.

### `generate_report.py`
- Usage: `python generate_report.py --artifacts artifacts/<run>`.
- Produces histogram and percentile charts if the relevant CSV files exist.
- Copies `output_diff.json` information (if present) into the final `report.json` for single-source reporting.

## Common issues & troubleshooting

- **`times.json`/`profile.json` structure differs:** The parsers try multiple schema variants, but if a new TensorRT version changes keys, inspect the JSON and extend the key lists (`LAT_KEYS`, `TIME_KEYS`, etc.) accordingly.
- **Dynamic shapes not resolved:** Supply explicit shapes via `--shape` when running `trtexec` and forward the same definition to `compare_model_outputs.py` through `--shapes`.
- **Engine benchmarking failures:** Confirm that the TensorRT engine supports the requested shape/profile combination and that the `trtexec` binary is visible on `$PATH`. For Jetson devices, ensure GPU clocks are locked to avoid watchdog timeouts.
- **Missing dependencies:** Install a CUDA-compatible `torch` build for PyTorch benchmarking and verify that your TensorRT installation ships a working `trtexec` executable.
- **Matplotlib backend errors:** All scripts set `matplotlib` to use the non-interactive `Agg` backend, so they can run on headless servers.

## Cleaning up

Artefacts are stored under the provided `--outdir`. Remove or archive completed runs manually, e.g.:

```bash
rm -rf artifacts/my_model
```

Feel free to adapt the scripts (e.g. extend CSV exports, integrate into CI pipelines, or add additional plots) to match your workflow.

# PyTorch Evaluation Toolkit

The PyTorch benchmarking utilities focus on measuring raw GPU inference throughput and latency for YOLOv8 Pose PyTorch models.

## Overview

- `bench_pt_yolo_pose.py` &ndash; Executes warmup + timed iterations of a YOLOv8 Pose `.pt` checkpoint on a CUDA device and reports
  host/H2D/GPU/D2H latency statistics alongside overall throughput. The script supports grayscale (1-channel) and RGB (3-channel)
  inputs, configurable batch sizes, FP32/FP16 precision via AMP, and optional TF32 disabling to mirror TensorRT behaviour.

## Prerequisites

1. Linux environment with a CUDA-capable GPU.
2. Python ≥ 3.8 with:
   - `torch` (compiled with CUDA support for your driver).
   - Optional: `ultralytics` if you wish to load checkpoints with the Ultralytics API via `--ultra`.
3. (Recommended) Create a Python virtual environment and install the dependencies, e.g.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install ultralytics  # optional, only needed when using --ultra
   ```

## Usage

Run the benchmark by pointing the CLI at a YOLOv8 Pose checkpoint and specifying the desired shape/precision options:

```bash
python evaluation_tool/bench_pt_yolo_pose.py \
  --pt /path/to/best.pt \
  --imgsz 640 --ch 1 --batch 1 \
  --iters 2000 --warmup 200 \
  --device cuda:0 --dtype fp32 --ultra
```

### Key arguments

- `--pt` (**required**): Path to the YOLOv8 Pose `.pt` weights file.
- `--imgsz`: Square input resolution (height = width); default `640`.
- `--ch`: Input channels (1 for grayscale, 3 for RGB); default `1`.
- `--batch`: Batch size used for both warmup and timed iterations; default `1`.
- `--iters`: Number of timed iterations included in the statistics; default `2000`.
- `--warmup`: Warmup iterations (excluded from reporting); default `200`.
- `--device`: CUDA device string (e.g. `cuda:0`).
- `--dtype`: Inference precision, `fp32` (default) or `fp16` (enables AMP autocast).
- `--no_tf32`: Disable TF32 matrix math (enabled by default for parity with TensorRT).
- `--ultra`: Attempt to load weights through `ultralytics.YOLO` before falling back to `torch.load`.

### Output

The script prints three sections to STDOUT:

1. **Model** &ndash; Resolved checkpoint path, selected device, input shape, dtype, and TF32 state.
2. **Performance summary** &ndash; Throughput in queries-per-second (QPS) and total host wall time in seconds.
3. **Latency breakdown** &ndash; Table of host, H2D, GPU, and D2H latency percentiles (min/max/mean/median/p90/p95/p99 in ms).

Additionally, two artefacts are written alongside the checkpoint:

- `bench_pt_result.csv` – CSV file mirroring the latency table for downstream plotting.
- `bench_pt_result.json` – JSON document capturing CLI arguments, throughput, wall time, and per-stage latency statistics.

Re-running the command with identical inputs should produce stable metrics (within ±5%). Use `--dtype fp16` to benchmark automatic mixed precision and append `--no_tf32` if TF32 should be disabled to match framework-specific baselines.
