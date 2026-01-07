# Building Triage Vision Android

This document explains how to build the complete Triage Vision app, including native ML libraries and models.

## Prerequisites

- **Android Studio** Arctic Fox or newer
- **Android SDK** API 34
- **Android NDK** 25.1.8937393 (installed via SDK Manager)
- **CMake** 3.22.1 (installed via SDK Manager)
- **JDK 17**
- **Git**

## Project Structure

```
app/src/main/
├── java/               # Kotlin source code
├── cpp/                # Native C++ code
│   ├── ncnn/           # NCNN library (not in git)
│   ├── llama.cpp/      # llama.cpp library (not in git)
│   ├── fast_pipeline/  # YOLO + depth processing
│   ├── slow_pipeline/  # SmolVLM inference
│   └── jni/            # JNI bridge
└── assets/
    └── models/         # ML model files (not in git)
```

## Step 1: Clone the Repository

```bash
git clone https://github.com/astinard/triage-vision-android.git
cd triage-vision-android
```

## Step 2: Set Up NCNN

NCNN is a high-performance neural network inference framework optimized for mobile.

### Option A: Download Pre-built (Recommended)

```bash
cd app/src/main/cpp

# Download NCNN Android Vulkan release
wget https://github.com/Tencent/ncnn/releases/download/20231027/ncnn-20231027-android-vulkan.zip
unzip ncnn-20231027-android-vulkan.zip
mv ncnn-20231027-android-vulkan ncnn
```

### Option B: Build from Source

```bash
cd app/src/main/cpp

# Clone NCNN
git clone https://github.com/Tencent/ncnn.git
cd ncnn

# Create build directory
mkdir build-android && cd build-android

# Configure for Android arm64-v8a with Vulkan
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DNCNN_VULKAN=ON \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF \
      ..

# Build
make -j$(nproc)

# Install to parent directory
make install DESTDIR=../install
```

## Step 3: Set Up llama.cpp

llama.cpp provides efficient LLM inference on edge devices.

```bash
cd app/src/main/cpp

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build for Android arm64-v8a
mkdir build-android && cd build-android

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DLLAMA_NATIVE=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      ..

make -j$(nproc)
```

## Step 4: Obtain Model Files

### YOLO11n for Person Detection

Convert YOLO11n to NCNN format:

```bash
# Install ultralytics
pip install ultralytics

# Export to NCNN
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='ncnn')

# This creates yolo11n_ncnn_model/ with:
# - yolo11n.ncnn.param
# - yolo11n.ncnn.bin
```

Copy to assets:
```bash
mkdir -p app/src/main/assets/models
cp yolo11n_ncnn_model/yolo11n.ncnn.* app/src/main/assets/models/
```

### SmolVLM-500M for Scene Understanding

Download and quantize SmolVLM:

```bash
# Clone and build llama.cpp tools (on host machine)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make

# Download SmolVLM-500M
pip install huggingface_hub
huggingface-cli download HuggingFaceTB/SmolVLM-500M-Instruct --local-dir smolvlm

# Convert to GGUF
python convert_hf_to_gguf.py smolvlm --outfile smolvlm-500m-f16.gguf

# Quantize to Q4_K_S for mobile (smaller, faster)
./llama-quantize smolvlm-500m-f16.gguf smolvlm-500m-q4_k_s.gguf Q4_K_S

# Copy to assets
cp smolvlm-500m-q4_k_s.gguf app/src/main/assets/models/
```

## Step 5: Build the App

### From Android Studio

1. Open the project in Android Studio
2. Sync Gradle
3. Build > Make Project

### From Command Line

```bash
# Set JAVA_HOME to JDK 17
export JAVA_HOME=/path/to/jdk17

# Build debug APK
./gradlew assembleDebug

# Build release APK
./gradlew assembleRelease

# Output: app/build/outputs/apk/
```

## Step 6: Install on Device

### Using ADB

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### For Mason Scan 600

The Mason Scan 600 uses standard ADB. Connect via USB and:

```bash
adb devices  # Verify device is connected
adb install -r app-debug.apk
```

## Troubleshooting

### NCNN not found
```
CMake Warning: NCNN not found at .../cpp/ncnn
```
Ensure NCNN is extracted to `app/src/main/cpp/ncnn/` with the correct structure.

### llama.cpp not found
```
CMake Warning: llama.cpp not found at .../cpp/llama.cpp
```
Ensure llama.cpp is cloned to `app/src/main/cpp/llama.cpp/`.

### Models not loading
Check logcat for model path issues:
```bash
adb logcat -s TriageVisionApp:* NativeBridge:*
```

Models should be in:
- `/data/data/com.triage.vision/files/models/` (internal)
- `/storage/emulated/0/Android/data/com.triage.vision/files/models/` (external)

### Native library crash
Ensure you're building for the correct ABI:
```bash
# Check device ABI
adb shell getprop ro.product.cpu.abi
# Should return: arm64-v8a
```

## Model Sizes

| Model | Format | Size | Purpose |
|-------|--------|------|---------|
| YOLO11n | NCNN | ~12 MB | Person detection, pose |
| SmolVLM-500M | GGUF Q4_K_S | ~350 MB | Scene understanding |

## Performance Targets

On Mason Scan 600 (Qualcomm QCM6490):

| Pipeline | Target FPS | Latency |
|----------|-----------|---------|
| Fast (YOLO) | 25-30 | <40ms |
| Fast + Depth | 20-28 | <50ms |
| Slow (SmolVLM) | 0.1-0.2 | 5-10s |

## CI/CD Notes

The GitHub Actions CI builds the Kotlin/Java layer but does **not** include:
- NCNN library
- llama.cpp library
- Model files

These are intentionally excluded from git (see `.gitignore`) due to size and licensing. CI verifies the app compiles but won't produce a functional APK without the native components.
