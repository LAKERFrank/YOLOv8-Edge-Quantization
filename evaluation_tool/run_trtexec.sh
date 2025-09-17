#!/usr/bin/env bash
# Wrapper around trtexec to standardise benchmark runs and collect JSON outputs.
set -euo pipefail

print_usage() {
  cat <<'USAGE'
Usage: $0 --engine <path.engine> [options]

Options:
  --engine <file>        TensorRT engine file to benchmark (required)
  --batch <N>            Batch size (default: 1)
  --iters <N>            Number of benchmark iterations (default: 200)
  --warmup <N>           Warmup iterations before measuring (default: 500)
  --avgRuns <N>          Number of averages for latency reported by trtexec (default: 100)
  --outdir <DIR>         Directory for artifacts (default: artifacts/<timestamp>)
  --useCudaGraph         Enable CUDA graph capture during benchmarking
  --shape <name:shape>   Dynamic shape specification passed to trtexec (can be repeated)
  --fp16                 Enable FP16 mode when running trtexec
  --int8                 Enable INT8 mode when running trtexec
  --trtexec <path>       Path to trtexec binary (default: trtexec from PATH)
  --extra "<args>"       Extra arguments passed directly to trtexec
  -h, --help             Show this message

Any additional arguments after "--" are forwarded to trtexec verbatim.
USAGE
}

BATCH=1
ITERS=200
WARMUP=500
USE_CUDA_GRAPH=0
OUTDIR="artifacts/$(date +%Y%m%d_%H%M%S)"
ENGINE=""
TRTEXEC="trtexec"
AVGRUNS=100
EXTRA_ARGS=()
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      ENGINE="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --iters|--iterations)
      ITERS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --avgRuns)
      AVGRUNS="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --useCudaGraph)
      USE_CUDA_GRAPH=1
      shift 1
      ;;
    --shape)
      EXTRA_ARGS+=("--shapes=$2")
      shift 2
      ;;
    --fp16|--int8|--best)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
    --trtexec)
      TRTEXEC="$2"
      shift 2
      ;;
    --extra)
      read -r -a extra_split <<< "$2"
      EXTRA_ARGS+=("${extra_split[@]}")
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      FORWARD_ARGS=("$@")
      break
      ;;
    *)
      echo "[run_trtexec] Unknown argument: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

if [[ -z "$ENGINE" ]]; then
  echo "[run_trtexec] Please provide --engine <engine_file>" >&2
  print_usage
  exit 1
fi

if [[ ! -f "$ENGINE" ]]; then
  echo "[run_trtexec] Engine file '$ENGINE' does not exist" >&2
  exit 1
fi

if ! command -v "$TRTEXEC" >/dev/null 2>&1; then
  echo "[run_trtexec] trtexec binary '$TRTEXEC' not found in PATH" >&2
  exit 2
fi

ENGINE_ABS=$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$ENGINE")
ENGINE_SIZE_BYTES=$(stat -c%s "$ENGINE_ABS")
ENGINE_SIZE_MIB=$(python3 -c 'import sys; print(float(sys.argv[1])/(1024**2))' "$ENGINE_SIZE_BYTES")
printf '[run_trtexec] Engine file: %s (%s bytes, %.2f MiB)\n' "$ENGINE_ABS" "$ENGINE_SIZE_BYTES" "$ENGINE_SIZE_MIB"

mkdir -p "$OUTDIR"
TIMES_JSON="$OUTDIR/times.json"
PROFILE_JSON="$OUTDIR/profile.json"
LOG_FILE="$OUTDIR/trtexec_stdout.log"
ENGINE_METADATA_JSON="$OUTDIR/engine_metadata.json"

python3 - "$ENGINE_ABS" "$ENGINE_METADATA_JSON" <<'PY'
import datetime as dt
import json
import os
import sys

engine_path = sys.argv[1]
out_path = sys.argv[2]

try:
    stat_result = os.stat(engine_path)
except OSError as exc:  # pragma: no cover - handled in shell prior, but kept for completeness
    raise SystemExit(f"Failed to stat engine file {engine_path}: {exc}")

info = {
    "path": engine_path,
    "filename": os.path.basename(engine_path),
    "size_bytes": int(stat_result.st_size),
    "size_megabytes": stat_result.st_size / (1024 ** 2),
    "modified_iso": dt.datetime.fromtimestamp(stat_result.st_mtime).astimezone().isoformat(),
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
PY

echo "[run_trtexec] Saved engine metadata to $ENGINE_METADATA_JSON"

CMD=("$TRTEXEC" "--loadEngine=$ENGINE" "--batch=$BATCH" "--warmUp=$WARMUP" "--iterations=$ITERS" "--avgRuns=$AVGRUNS" "--exportTimes=$TIMES_JSON" "--exportProfile=$PROFILE_JSON" "--dumpProfile")

if [[ "$USE_CUDA_GRAPH" -eq 1 ]]; then
  CMD+=("--useCudaGraph")
fi

CMD+=("${EXTRA_ARGS[@]}")
if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
  CMD+=("${FORWARD_ARGS[@]}")
fi

{
  echo "[run_trtexec] Running trtexec command:"
  printf '  %q' "${CMD[@]}"
  echo
} | tee "$LOG_FILE"

if ! "${CMD[@]}" &>>"$LOG_FILE"; then
  echo "[run_trtexec] trtexec execution failed. See $LOG_FILE for details." >&2
  exit 3
fi

echo "[run_trtexec] trtexec finished successfully. Artifacts stored in $OUTDIR"
