# TensorRT `trtexec` Evaluation Toolkit

This folder contains a small toolkit for running reproducible TensorRT engine benchmarks using `trtexec`, parsing the generated artefacts, comparing engine outputs, and building presentation-ready charts/reports.

## Directory layout

- `run_trtexec.sh` &ndash; Bash wrapper that standardises `trtexec` runs and stores JSON artefacts inside an output folder.
- `parse_trtexec_times.py` &ndash; Parses `times.json` produced via `--exportTimes`, computes latency / throughput statistics, and exports summary CSV/JSON files.
- `parse_trtexec_profile.py` &ndash; Parses per-layer timing information from `profile.json` and produces CSV + bar-chart visualisations.
- `compare_model_outputs.py` &ndash; Executes two engines (e.g. FP32 vs INT8) with identical random inputs using TensorRT + PyCUDA and reports element-wise differences.
- `generate_report.py` &ndash; Aggregates the artefacts into a consolidated report (`report.json`/`report.md`) and generates latency/per-layer plots.

> **Note:** Python scripts expect Python ≥ 3.8. Optional features require additional dependencies listed below.

## Prerequisites

1. Linux environment with TensorRT installed and `trtexec` accessible via `$PATH`.
2. Python ≥ 3.8 with the following packages:
   - Required: `numpy`, `pandas`, `matplotlib`.
   - Optional: `scipy`, `tqdm` for extended analyses (not required by default scripts).
   - `compare_model_outputs.py` additionally requires `tensorrt` and `pycuda` Python bindings.
3. (Recommended) Create a Python virtual environment and install the dependencies, e.g.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas matplotlib pycuda tensorrt
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
   The script saves `times.json`, `profile.json`, and a `trtexec_stdout.log` file into the specified `--outdir` (created automatically).

2. **Parse latency statistics**
   ```bash
   python evaluation_tool/parse_trtexec_times.py artifacts/my_model/times.json 1
   ```
   Outputs:
   - `summary.json` / `summary.csv` containing throughput and latency statistics (min/mean/median/percentiles).
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

5. **(Optional) Compare FP32 vs INT8 outputs**
   ```bash
   python evaluation_tool/compare_model_outputs.py \
     --ref_engine path/to/fp32.engine \
     --test_engine path/to/int8.engine \
     --batch 1 \
     --n 10 \
     --outdir artifacts/compare_fp32_int8
   ```
   The script executes both engines with identical random inputs and stores:
   - `output_diff.json` containing aggregated MAE / maximum absolute differences per output tensor.
   - `per_sample_diff.json` with raw per-sample metrics.
   - Per-sample inputs/outputs inside `artifacts/compare_fp32_int8/samples/` (disable with `--skip-output-save`).
   - Use `--input-shape name:dim,dim,...` for dynamic shapes. Provide the full explicit shape (include batch dimension for implicit-batch engines).

## Script details & options

### `run_trtexec.sh`
- Accepts `--shape <tensor:dimx...>` to forward dynamic shape definitions to `trtexec`.
- Use `--extra "--separateProfileRun --memPoolSize=workspace:4096"` or append arguments after `--` to pass additional flags directly.
- Logs the exact command and all `trtexec` output to `<outdir>/trtexec_stdout.log`.

### `parse_trtexec_times.py`
- Usage: `python parse_trtexec_times.py <times.json> <batch> [outdir]`.
- Robust to minor schema changes (`times`, `perIteration`, nested dicts) and automatically converts units to milliseconds.
- Produces `latency_percentiles.csv` for plotting percentile curves.

### `parse_trtexec_profile.py`
- Usage: `python parse_trtexec_profile.py <profile.json> [--outdir DIR] [--topk N]`.
- Handles multiple nesting layouts (`layers`, `profile`, generic nested dicts) and deduplicates repeated entries by `(name, avg_time)` pairs.

### `compare_model_outputs.py`
- Requires both engines to expose identical bindings (names and dtypes). The script validates compatibility before running.
- Supports explicit and implicit batch engines. For implicit batch networks, ensure `--batch` matches the engine batch size.
- Use `--profile-index` to select an optimisation profile when running engines built with multiple profiles.
- If any input has dynamic dimensions, specify them via `--input-shape input_name:dim,dim,...`. For example: `--input-shape images:1,3,640,640`.

### `generate_report.py`
- Usage: `python generate_report.py --artifacts artifacts/<run>`.
- Produces histogram and percentile charts if the relevant CSV files exist.
- Copies `output_diff.json` information (if present) into the final `report.json` for single-source reporting.

## Common issues & troubleshooting

- **`times.json`/`profile.json` structure differs:** The parsers try multiple schema variants, but if a new TensorRT version changes keys, inspect the JSON and extend the key lists (`LAT_KEYS`, `TIME_KEYS`, etc.) accordingly.
- **Dynamic shapes not resolved:** Supply explicit shapes via `--shape` when running `trtexec` and use the same shapes with `--input-shape` for output comparison.
- **Engine execution failures in comparison script:** Confirm that both engines were built with identical input bindings and that the active optimisation profile supports the requested shapes. For Jetson devices, ensure GPU clocks are locked to avoid watchdog timeouts.
- **Missing dependencies:** Install `pycuda` and TensorRT Python wheels matching your TensorRT version. On Jetson, these are typically pre-installed in `/usr/lib/python*/dist-packages`.
- **Matplotlib backend errors:** All scripts set `matplotlib` to use the non-interactive `Agg` backend, so they can run on headless servers.

## Cleaning up

Artefacts are stored under the provided `--outdir`. Remove or archive completed runs manually, e.g.:

```bash
rm -rf artifacts/my_model
```

Feel free to adapt the scripts (e.g. extend CSV exports, integrate into CI pipelines, or add additional plots) to match your workflow.
