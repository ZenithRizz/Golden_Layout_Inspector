#!/usr/bin/env bash
# Install ExecuTorch from source (needed for export AND for building the llama runner).
# Run on: your Ubuntu laptop (for export) and on the Raspberry Pi 5 (to build llama_main).
# Takes 20-40 min on a laptop, longer on the Pi.  Needs: git, cmake>=3.26, g++, python3.10-3.12.
set -euo pipefail
unset PYTHONPATH                         # keep ROS (or other) Python paths out of the build
ET_DIR="${ET_DIR:-$HOME/executorch}"
ET_REF="${ET_REF:-release/1.0}"          # pin a release branch; change if the docs you follow use another

sudo apt-get update
sudo apt-get install -y git cmake build-essential python3-venv python3-dev
python3 -m venv "$HOME/et-venv"
source "$HOME/et-venv/bin/activate"
pip install -U pip
pip install "cmake>=3.29" ninja huggingface_hub   # Ubuntu 22.04's apt cmake (3.22) is too old

if [ ! -d "$ET_DIR" ]; then
  git clone -b "$ET_REF" --recurse-submodules https://github.com/pytorch/executorch.git "$ET_DIR"
fi
cd "$ET_DIR"
git submodule sync && git submodule update --init --recursive
./install_executorch.sh                 # installs the executorch Python package + deps
./examples/models/llama/install_requirements.sh
echo "ExecuTorch ready in $ET_DIR (venv: source ~/et-venv/bin/activate)"
