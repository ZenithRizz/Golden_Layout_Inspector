#!/usr/bin/env bash
unset PYTHONPATH
# Check Llama runs through ExecuTorch and prints timing stats (PyTorchObserver line).
set -euo pipefail
ET_DIR="${ET_DIR:-$HOME/executorch}"
M="${M:-$HOME/gli-models/llama}"
"$ET_DIR/cmake-out/examples/models/llama/llama_main" \
  --model_path="$M/llama3_2_1b_instruct_8da4w_xnn.pte" \
  --tokenizer_path="$M/tokenizer.model" \
  --cpu_threads=4 --temperature=0 --seq_len=256 \
  --prompt="<|start_header_id|>user<|end_header_id|>

In one sentence, what is a March C- memory test?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
