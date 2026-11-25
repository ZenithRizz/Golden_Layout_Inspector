#!/usr/bin/env bash
# Start the inspection station.   bash scripts/run_station.sh [source]
#   source: virtual (default) | usb:0 | picam | http://127.0.0.1:8080 (phone IP Webcam) | path/to/image.jpg
# Works on: Raspberry Pi / Arm Linux, Android (Termux), and the x86 laptop for development.
# Llama is used automatically when the runner, .pte and tokenizer are found (override with env vars):
#   GLI_LLAMA_RUNNER, GLI_LLAMA_PTE, GLI_LLAMA_TOKENIZER, GLI_LLAMA_THREADS
# View the UI from another device on the same Wi-Fi/hotspot:  GLI_HOST=0.0.0.0 bash scripts/run_station.sh ...
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
SRC="${1:-virtual}"
M="$HOME/gli-models/llama"

# Llama runner: phone copy (Termux) first, then the ExecuTorch build tree (laptop / Pi)
if [ -z "${GLI_LLAMA_RUNNER:-}" ]; then
  if [ -x "$M/llama_main" ]; then GLI_LLAMA_RUNNER="$M/llama_main"
  else GLI_LLAMA_RUNNER="$HOME/executorch/cmake-out/examples/models/llama/llama_main"; fi
fi
: "${GLI_LLAMA_PTE:=$M/llama3_2_1b_instruct_8da4w_xnn.pte}"
: "${GLI_LLAMA_TOKENIZER:=$M/tokenizer.model}"
: "${GLI_LLAMA_THREADS:=4}"
export GLI_LLAMA_RUNNER GLI_LLAMA_PTE GLI_LLAMA_TOKENIZER GLI_LLAMA_THREADS

# Android/Termux: keep the CPU awake while the station runs
if [ -n "${TERMUX_VERSION:-}" ] && command -v termux-wake-lock >/dev/null; then termux-wake-lock || true; fi

# FastAPI server if available, otherwise the dependency-free server (same UI and API)
if python -c "import fastapi, uvicorn, multipart" >/dev/null 2>&1; then SERVER=gli.server; else SERVER=gli.server_lite; fi

HEF_ARG=()
[ -f models/defect_cls.hef ] && HEF_ARG=(--hef models/defect_cls.hef)
exec python -m "$SERVER" --design designs/sram16x16_scn4m --source "$SRC" "${HEF_ARG[@]}" \
     --llama-threads "$GLI_LLAMA_THREADS" --host "${GLI_HOST:-127.0.0.1}" --port "${GLI_PORT:-8000}"
