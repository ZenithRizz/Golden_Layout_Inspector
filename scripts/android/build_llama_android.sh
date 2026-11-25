#!/usr/bin/env bash
# Cross-compile ExecuTorch's llama_main for an Android phone (arm64-v8a), on the Ubuntu/WSL laptop.
# Follows ExecuTorch release/1.0 examples/models/llama/README.md "Step 4: Run benchmark on Android phone".
# Prereq: 01_setup_executorch.sh done (~/executorch + ~/et-venv).  Output: ~/gli-models/android/llama_main
set -euo pipefail
unset PYTHONPATH
source "$HOME/et-venv/bin/activate"
ET_DIR="${ET_DIR:-$HOME/executorch}"
JOBS="${JOBS:-2}"
NDK_VER="${NDK_VER:-r27b}"                       # version tested by ExecuTorch
export ANDROID_NDK="${ANDROID_NDK:-$HOME/android-ndk-$NDK_VER}"
OUT="$HOME/gli-models/android"
mkdir -p "$OUT"

# 1) Android NDK (about 700 MB download, 2 GB unpacked) - skipped if already present
if [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
  sudo apt-get install -y unzip wget
  cd "$HOME"
  wget -q --show-progress "https://dl.google.com/android/repository/android-ndk-$NDK_VER-linux.zip"
  unzip -q "android-ndk-$NDK_VER-linux.zip"
  rm -f "android-ndk-$NDK_VER-linux.zip"
fi
TC="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
ANDROID_ARGS=(-DCMAKE_TOOLCHAIN_FILE="$TC" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DANDROID_STL=c++_static)

cd "$ET_DIR"
rm -rf cmake-out-android/CMakeCache.txt cmake-out-android/CMakeFiles

# 2) ExecuTorch core + LLM libraries for Android.  Prefer the 'llm' preset (sets every required option);
#    fall back to the README's explicit flags (+ NAMED_DATA_MAP, required by 1.0's option checks).
if ! cmake --preset llm -B cmake-out-android "${ANDROID_ARGS[@]}" \
      -DCMAKE_INSTALL_PREFIX=cmake-out-android -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=python; then
  rm -rf cmake-out-android/CMakeCache.txt cmake-out-android/CMakeFiles
  cmake "${ANDROID_ARGS[@]}" -DCMAKE_INSTALL_PREFIX=cmake-out-android -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON -DEXECUTORCH_ENABLE_LOGGING=1 -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -Bcmake-out-android .
fi
cmake --build cmake-out-android -j"$JOBS" --target install --config Release

# 3) llama runner for Android
cmake "${ANDROID_ARGS[@]}" -DCMAKE_INSTALL_PREFIX=cmake-out-android -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=python -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_LLM=ON -DSUPPORT_REGEX_LOOKAHEAD=ON \
  -DBUILD_TESTING=OFF \
  -Bcmake-out-android/examples/models/llama examples/models/llama
cmake --build cmake-out-android/examples/models/llama -j"$JOBS" --config Release

cp cmake-out-android/examples/models/llama/llama_main "$OUT/llama_main"
file "$OUT/llama_main" || true
ls -lh "$OUT/llama_main"
echo "Android runner ready: $OUT/llama_main   (should say: ELF 64-bit LSB ... ARM aarch64)"
