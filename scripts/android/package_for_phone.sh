#!/usr/bin/env bash
# On the laptop: bundle everything the phone needs into one folder you can copy to the phone.
#   ~/gli-phone/gli.zip, llama_main (Android build), the .pte and tokenizer.model
set -euo pipefail
OUT="$HOME/gli-phone"
mkdir -p "$OUT"
cd "$HOME"
rm -f "$OUT/gli.zip"
zip -qr "$OUT/gli.zip" gli -x "gli/.venv/*" "gli/runs/*" "gli/bench/*" "gli/data/*/crops.npz" \
    "gli/models/calib_nhwc.npy" "gli/models/defect_cls.pt" "gli/**/__pycache__/*"
cp "$HOME/gli-models/android/llama_main" "$OUT/" 2>/dev/null || echo "WARNING: Android llama_main not built yet (build_llama_android.sh)"
cp "$HOME/gli-models/llama/llama3_2_1b_instruct_8da4w_xnn.pte" "$HOME/gli-models/llama/tokenizer.model" "$OUT/" 2>/dev/null \
   || echo "WARNING: .pte/tokenizer not found in ~/gli-models/llama"
ls -lh "$OUT"
echo
echo "Copy the contents of $OUT to the phone's Download folder (USB cable 'File transfer' mode, or Google Drive)."
echo "Windows path to this folder:  \\\\wsl\$\\Ubuntu$OUT"
