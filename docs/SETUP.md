# Triage Vision Android - Setup Guide

Complete setup guide for building and deploying Triage Vision to the Mason Scan 600.

## Prerequisites

### Development Environment

| Requirement | Version | Notes |
|-------------|---------|-------|
| Android Studio | Latest (Hedgehog+) | With Kotlin 1.9+ |
| Android SDK | 34 | Target SDK |
| Android NDK | r25+ | For native builds |
| CMake | 3.22+ | Via SDK Manager |
| JDK | 17+ | Required for Gradle |
| Git | Latest | For cloning deps |

### Target Device

| Spec | Mason Scan 600 |
|------|----------------|
| OS | Android 13 |
| CPU | Qualcomm QCM6490 |
| GPU | Adreno 643 |
| NPU | 10 TOPS Hexagon |
| RAM | 6GB |
| ABI | arm64-v8a |

## Quick Start

```bash
# 1. Clone the project
git clone <repo-url> triage-vision-android
cd triage-vision-android

# 2. Build native libraries
./scripts/build_native_libs.sh
# Select option 5 (Quick setup)

# 3. Download ML models
./scripts/download_models.sh

# 4. Open in Android Studio and build
```

## Detailed Setup

### Step 1: Install Android NDK

```bash
# Via Android Studio SDK Manager:
# Tools > SDK Manager > SDK Tools > NDK (Side by side)

# Or via command line:
sdkmanager "ndk;25.2.9519653"

# Set environment variable
export ANDROID_NDK="$HOME/Library/Android/sdk/ndk/25.2.9519653"
```

### Step 2: Build Native Libraries

#### Option A: Use Pre-built NCNN + Build llama.cpp

```bash
./scripts/build_native_libs.sh
# Select option 5
```

#### Option B: Build Everything from Source

```bash
./scripts/build_native_libs.sh
# Select option 4
```

This will:
1. Clone and build NCNN with Vulkan support
2. Clone and build llama.cpp with OpenCL support
3. Install libraries to `app/src/main/cpp/`

### Step 3: Download ML Models

```bash
./scripts/download_models.sh
```

Models required:
- `smolvlm-500m-q4_k_s.gguf` (~293MB) - VLM for scene understanding
- `yolo11n.ncnn.bin/param` (~5MB) - Fast detection

#### Manual Model Download

**SmolVLM (GGUF):**
```bash
# Download from HuggingFace
curl -L -o app/src/main/assets/models/smolvlm-500m-q4_k_s.gguf \
    "https://huggingface.co/mradermacher/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct.Q4_K_S.gguf"
```

**YOLO11n (NCNN):**
```python
# Export from Ultralytics
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='ncnn')
# Copy yolo11n_ncnn_model/* to app/src/main/assets/models/
```

### Step 4: Configure Android Studio

1. Open `triage-vision-android` in Android Studio
2. Sync Gradle (should auto-detect NDK)
3. If CMake errors: File > Sync Project with Gradle Files

### Step 5: Build APK

```bash
# Debug build
./gradlew assembleDebug

# Release build
./gradlew assembleRelease
```

### Step 6: Deploy to Mason Scan 600

```bash
# Connect device via USB
adb devices

# Install
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Or deploy via Android Studio
```

## Configuration

### Monitoring Settings

Edit `app/src/main/res/values/config.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <!-- Fast Pipeline -->
    <integer name="fast_pipeline_fps">15</integer>
    <integer name="stillness_alert_seconds">1800</integer>
    <bool name="fall_detection_enabled">true</bool>

    <!-- Slow Pipeline (VLM) -->
    <integer name="vlm_interval_minutes">15</integer>
    <bool name="vlm_trigger_on_stillness">true</bool>
    <bool name="vlm_trigger_on_fall">true</bool>

    <!-- Data Retention (HIPAA) -->
    <integer name="data_retention_hours">24</integer>
    <bool name="auto_purge_enabled">true</bool>
</resources>
```

### VLM Prompt Customization

Edit `SlowPipeline.kt` `DEFAULT_PROMPT` for your specific use case:

```kotlin
const val DEFAULT_PROMPT = """
Analyze this patient monitoring image. Describe:
1. Patient position...
[customize for your clinical needs]
"""
```

## Troubleshooting

### Native Library Errors

**UnsatisfiedLinkError:**
```
Make sure:
1. NDK ABI matches device (arm64-v8a for Mason Scan 600)
2. All .so files are in app/src/main/jniLibs/arm64-v8a/
3. System.loadLibrary() matches library names
```

**CMake Build Failures:**
```bash
# Clean and rebuild
cd native_build && rm -rf * && cd ..
./scripts/build_native_libs.sh
```

### Model Loading Errors

**Model not found:**
```
Check paths:
1. Models in app/src/main/assets/models/
2. Or in device external storage: /sdcard/Android/data/com.triage.vision/files/models/
```

**Out of memory:**
```
Use smaller quantization:
- Try Q3_K_S instead of Q4_K_S
- Reduce VLM interval to avoid concurrent loads
```

### Camera Issues

**Camera permission denied:**
```
1. Check AndroidManifest.xml has CAMERA permission
2. Grant permission in device Settings > Apps > Triage Vision
3. For Android 13+, ensure foreground service type is "camera"
```

**Black preview:**
```
1. Check CameraX lifecycle binding
2. Ensure PreviewView is properly sized
3. Test with default camera app first
```

## Performance Optimization

### Battery Life

For 8+ hour monitoring:
- Fast pipeline: 15 FPS (not 30)
- VLM interval: 15+ minutes
- Disable Vulkan if GPU power draw is high

### Inference Speed

| Model | Expected Latency |
|-------|------------------|
| YOLO11n (NCNN/Vulkan) | <50ms |
| SmolVLM-500M (Q4) | 2-5 sec |

### Memory Management

- Models loaded on-demand, unloaded when idle
- Frame buffer recycling in CameraX
- 24-hour data purge for observation DB

## Security Considerations

### HIPAA Compliance

This app is designed for HIPAA compliance:

1. **No cloud connectivity** - All inference on-device
2. **No data export** - Unless manually triggered
3. **Auto-purge** - 24-hour data retention default
4. **Encrypted storage** - Room DB + SharedPreferences
5. **No backup** - `android:allowBackup="false"`

### Additional Hardening

For production deployment:
- Enable ProGuard/R8 obfuscation
- Use Android Keystore for encryption keys
- Implement user authentication
- Add audit logging

## Support

For issues, check:
1. This documentation
2. Android Studio Logcat (filter: `TriageVision`)
3. GitHub Issues (if applicable)
