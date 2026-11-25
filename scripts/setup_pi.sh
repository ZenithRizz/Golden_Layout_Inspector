#!/usr/bin/env bash
# One-time setup on Raspberry Pi 5 (Raspberry Pi OS 64-bit, Bookworm or later).
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libgl1 v4l-utils
python3 -m venv --system-site-packages .venv     # system site packages: picamera2 if you use a Pi camera
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Performance governor for stable benchmarks (reverts on reboot)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null || true
v4l2-ctl --list-devices || true
echo "Done. Start with:  ./scripts/run_station.sh usb:0"
