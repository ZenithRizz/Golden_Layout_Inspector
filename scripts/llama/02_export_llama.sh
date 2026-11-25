#!/usr/bin/env bash
# Export Llama 3.2 1B Instruct to an ExecuTorch .pte (XNNPACK, 8-bit dynamic activations / 4-bit weights).
# Run on the Ubuntu laptop (x86 is fine: the .pte runs on the Pi's Arm CPU too).
# Prereq: HF access to meta-llama/Llama-3.2-1B-Instruct approved; `hf auth login` done; 01_setup_executorch.sh done.
set -euo pipefail
unset PYTHONPATH
source "$HOME/et-venv/bin/activate"
ET_DIR="${ET_DIR:-$HOME/executorch}"
OUT="${OUT:-$HOME/gli-models/llama}"
NAME="llama3_2_1b_instruct_8da4w_xnn"
mkdir -p "$OUT"

# 1) Meta-format weights: consolidated.00.pth, params.json, tokenizer.model (about 2.5 GB)
if [ ! -f "$OUT/hf/original/consolidated.00.pth" ]; then
  hf download meta-llama/Llama-3.2-1B-Instruct --include "original/*" --local-dir "$OUT/hf"
fi
CK="$OUT/hf/original"
cp "$CK/tokenizer.model" "$OUT/tokenizer.model"

# 2) Export config (format from the ExecuTorch "Exporting LLMs" docs)
CFG="$OUT/export_config.yaml"
cat > "$CFG" <<YAML
base:
  model_class: llama3_2
  checkpoint: $CK/consolidated.00.pth
  params: $CK/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: True
export:
  max_seq_length: 1024
  max_context_length: 1024
  output_dir: $OUT
  output_name: $NAME.pte
quantization:
  qmode: 8da4w
  group_size: 32
  embedding_quantize: 4,32
backend:
  xnnpack:
    enabled: True
    extended_ops: True
YAML

cd "$ET_DIR"
START=$(date +%s)
if ! python -m extension.llm.export.export_llm --config "$CFG"; then
  echo ">>> export_llm failed; trying the legacy export_llama entry point"
  python -m examples.models.llama.export_llama \
    --model llama3_2 -c "$CK/consolidated.00.pth" -p "$CK/params.json" \
    -kv --use_sdpa_with_kv_cache -X -qmode 8da4w --group_size 32 -d fp32 \
    --embedding-quantize 4,32 --max_seq_length 1024 --max_context_length 1024 \
    --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
    --output_name "$OUT/$NAME.pte"
fi

# 3) Find the .pte even if the exporter ignored the requested name/location
if [ ! -f "$OUT/$NAME.pte" ]; then
  NEW=$(find "$ET_DIR" "$OUT" -name "*.pte" -newermt "@$START" 2>/dev/null | head -1 || true)
  [ -n "$NEW" ] && mv "$NEW" "$OUT/$NAME.pte"
fi
ls -lh "$OUT"/*.pte "$OUT/tokenizer.model"
echo "OK: $OUT/$NAME.pte"
