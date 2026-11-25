#!/usr/bin/env bash
# Build ExecuTorch's llama_main runner (C++) - commands from examples/models/llama/README.md
# of ExecuTorch release/1.0 ("Step 3: Run on your computer to validate").
# Run natively on the Raspberry Pi 5 for the demo, and on the laptop for development.
set -euo pipefail
unset PYTHONPATH
source "$HOME/et-venv/bin/activate"
ET_DIR="${ET_DIR:-$HOME/executorch}"
JOBS="${JOBS:-2}"                         # keep low on 8-16 GB RAM machines
cd "$ET_DIR"

# clear any half-configured cache from earlier attempts
rm -rf cmake-out/CMakeCache.txt cmake-out/CMakeFiles

# 1) ExecuTorch core + LLM libraries (preset 'llm' turns on XNNPACK, quantized/optimized/LLM kernels,
#    module, tensor, named-data-map and LLM-runner extensions)
cmake --preset llm -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=cmake-out -DPYTHON_EXECUTABLE=python
cmake --build cmake-out -j"$JOBS" --target install --config Release

# 2) The llama runner
cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -Bcmake-out/examples/models/llama \
      examples/models/llama
cmake --build cmake-out/examples/models/llama -j"$JOBS" --config Release

ls -lh cmake-out/examples/models/llama/llama_main
echo "Runner: $ET_DIR/cmake-out/examples/models/llama/llama_main"
