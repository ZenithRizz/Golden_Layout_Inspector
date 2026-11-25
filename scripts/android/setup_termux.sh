#!/data/data/com.termux/files/usr/bin/bash
# Run INSIDE TERMUX on the phone, after copying gli.zip, llama_main, the .pte and tokenizer.model
# to the phone's Download folder.   bash ~/storage/downloads/setup_termux.sh  (or from the unzipped repo)
set -euo pipefail
DL="$HOME/storage/downloads"
if [ ! -d "$DL" ]; then
  echo "Granting Termux access to phone storage - tap ALLOW on the popup, then run this script again."
  termux-setup-storage
  exit 0
fi
pkg update -y
pkg install -y x11-repo                         # OpenCV for Python lives in Termux's x11-repo
pkg update -y
pkg install -y python python-numpy unzip
pkg install -y termux-api || true
pkg install -y opencv-python || pkg install -y opencv
python -c "import numpy, cv2; print('numpy', numpy.__version__, '| opencv', cv2.__version__); cv2.aruco.ArucoDetector" \
  || { echo "OpenCV in Termux lacks ArUco - tell Claude"; exit 1; }

# repo
if [ -f "$DL/gli.zip" ]; then
  rm -rf "$HOME/gli" && cd "$HOME" && unzip -q "$DL/gli.zip"
fi
# Llama runner + model (binaries can't run from shared storage, so copy into Termux's home)
mkdir -p "$HOME/gli-models/llama"
for f in llama_main llama3_2_1b_instruct_8da4w_xnn.pte tokenizer.model; do
  [ -f "$DL/$f" ] && cp "$DL/$f" "$HOME/gli-models/llama/"
done
[ -f "$HOME/gli-models/llama/llama_main" ] && chmod +x "$HOME/gli-models/llama/llama_main"
ls -lh "$HOME/gli-models/llama/" || true
echo
echo "Setup done.  Start the station with:"
echo "   cd ~/gli && bash scripts/run_station.sh http://USER:PASS@127.0.0.1:8080   (your IP Webcam login)"
