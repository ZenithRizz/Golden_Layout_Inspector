# Running the station on an Android phone

PS5 allows "Arm-based Android devices with camera". On the phone, everything runs on-device:

```
 Phone rear camera ─► IP Webcam app (http://127.0.0.1:8080, on the same phone, no internet)
                         │
 Termux (Linux on Android, Arm CPU):
   VISION PIPELINE   registration ─► golden diff ─► CNN (OpenCV DNN on the Arm CPU)
   ADVISOR PIPELINE  llama_main (ExecuTorch, XNNPACK/KleidiAI) ─► Llama 3.2 1B Instruct, 4-bit
   mapping · rules · SQLite log · web UI on http://127.0.0.1:8000
                         │
 Chrome on the phone (or a laptop browser on the phone's hotspot) shows the UI
```

Differences from the Pi build (state these in the report):
- The classifier runs in OpenCV's DNN module instead of ONNX Runtime (same model, outputs
  identical to within 2e-6).
- The web server is the dependency-free `gli/server_lite.py` (same UI and API).

## A. On the laptop (Ubuntu/WSL)

1. **Llama model:** finish the Colab export and put the `.pte` and `tokenizer.model` in `~/gli-models/llama/`.
2. **Android Llama runner** (downloads the Android NDK, about 700 MB; build takes 30–60 min):
   ```bash
   JOBS=2 bash ~/gli/scripts/android/build_llama_android.sh 2>&1 | tee ~/android_build.log
   ```
   It ends with `Android runner ready: /home/<you>/gli-models/android/llama_main`.
3. **Bundle everything for the phone:**
   ```bash
   bash ~/gli/scripts/android/package_for_phone.sh
   ```
   This creates `~/gli-phone/` containing `gli.zip`, `llama_main`, the `.pte` and `tokenizer.model`.
   In Windows Explorer the folder is at `\\wsl$\Ubuntu\home\<you>\gli-phone`.
4. **Copy those 4 files to the phone's Download folder**, plus `scripts/android/setup_termux.sh`
   (also in `gli.zip`, but easier to have loose). Use a USB cable in "File transfer" mode, or Google Drive.

## B. On the phone

1. Install **Termux from F-Droid** (f-droid.org → Termux), *not* the Play Store version.
2. Install **IP Webcam** (Play Store, by Pavel Khlebovich). Open it and set:
   - Video preferences → Video resolution: **1920×1080**
   - Focus mode: **continuous** (or tap to focus in the preview)
   - **Connection settings → Login/password:** set a username and password, so nobody else on
     the same Wi-Fi can view your camera. Avoid `@ : /` in the password.
   - Scroll down → **Start server**. Leave it running; you can switch apps.
   - When you're finished, open IP Webcam and press **Stop server**.
3. Open **Termux** and run:
   ```bash
   termux-setup-storage
   ```
   Tap **Allow**, then:
   ```bash
   bash ~/storage/downloads/setup_termux.sh
   ```
   This takes about 5–10 min (it installs Python, NumPy and OpenCV, then unzips the repo and copies the model files).
4. Start the station, putting your IP Webcam login in the address:
   ```bash
   cd ~/gli && bash scripts/run_station.sh http://USER:PASS@127.0.0.1:8080
   ```
   If the password has special characters, pass it through environment variables instead:
   ```bash
   GLI_CAM_USER=USER GLI_CAM_PASS='PASS' bash scripts/run_station.sh http://127.0.0.1:8080
   ```
   The password is never shown in the UI or saved in the inspection log.

   If the live view is turned 90° (IP Webcam sends the sensor's landscape image even when the
   phone is upright), add `GLI_CAM_ROTATE=90`, or `270` if it ends up upside down:
   ```bash
   GLI_CAM_ROTATE=90 bash scripts/run_station.sh http://USER:PASS@127.0.0.1:8080
   ```
   Inspection works at any rotation (the markers handle it); this only fixes the display.
   Or try it first without the camera: `bash scripts/run_station.sh virtual`.
5. Open **Chrome** on the phone at `http://127.0.0.1:8000`.

**Viewing on the laptop (better for recording the demo):**
1. Turn on the phone's **hotspot** and connect the laptop to it.
2. Start the station with `GLI_HOST=0.0.0.0 bash scripts/run_station.sh http://127.0.0.1:8080`.
3. Find the phone's IP with `ifconfig 2>/dev/null | grep "inet "` in Termux (usually `192.168.x.1`).
4. Open `http://<that IP>:8000` on the laptop.

No internet is used at any point.

## C. Physical setup

- Print `designs/sram16x16_scn4m/target_A4.pdf` in colour at 100%.
- Hold the phone 25–35 cm above the target, parallel to it. A phone stand, a stack of books or a glass sheet works well. All 4 corner markers must be in view.
- Light it evenly and avoid glare (matte paper helps).

## D. Benchmarks on the phone (for the report)

```bash
cd ~/gli
python scripts/benchmark.py --dies 20 \
  --llama-runner ~/gli-models/llama/llama_main \
  --llama-pte ~/gli-models/llama/llama3_2_1b_instruct_8da4w_xnn.pte \
  --llama-tokenizer ~/gli-models/llama/tokenizer.model
```

Results go to `bench/<stamp>/summary.md`. Record the phone model, SoC and RAM in the report.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pkg` errors / mirrors | `termux-change-repo`, pick a different mirror, retry |
| `Permission denied` running llama_main | it must be in `~/gli-models/llama/` (not Download); `chmod +x ~/gli-models/llama/llama_main` |
| Camera unavailable | IP Webcam server not started, or Android stopped it: reopen IP Webcam → Start server |
| Station stops when the screen turns off | the run script takes a wake lock; also exclude Termux and IP Webcam from battery optimisation (Settings → Apps → Battery → Unrestricted) |
| Llama killed / slow | close other apps; try `GLI_LLAMA_THREADS=4` or `2` |
