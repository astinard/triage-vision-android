# Triage Vision Android

On-device nursing triage vision system for Mason Scan 600. HIPAA-compliant patient monitoring with auto-charting.

## Features

- **Fast CLIP Classification**: MobileCLIP-S1 provides real-time (~200ms) patient state classification
- **Continuous Motion Monitoring**: YOLO11n detects patient movement, stillness, falls
- **Intelligent Scene Analysis**: SmolVLM-500M provides natural language observations
- **Auto-Charting**: Generates structured nursing notes in HL7 FHIR format
- **Fully On-Device**: No data leaves the device - HIPAA compliant by design

## Target Hardware

- **Device**: Mason Scan 600
- **Processor**: Qualcomm QCM6490 (10 TOPS NPU)
- **RAM**: 6GB
- **Camera**: 16MP + ToF depth sensor
- **OS**: Android 13

## Architecture

```
Camera → Fast Pipeline (MobileCLIP-S1/ONNX) → Real-time Classification
              ↓                                  (position, alertness, activity, comfort, safety)
         YOLO11n/NCNN → Motion/Pose Detection
              ↓ (triggers on events)
         Slow Pipeline (SmolVLM/llama.cpp) → Detailed Scene Understanding
              ↓
         Auto-Charting Engine → HL7 FHIR Output
```

### Backend Abstraction

The app uses a pluggable backend system for hardware acceleration:

```
┌─────────────────────────────────────┐
│         VisionBackend Interface      │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│          BackendRegistry             │
│  Auto-selects: QNN → NNAPI → CPU    │
└────┬────────────┬───────────────────┘
     │            │
┌────▼────┐ ┌─────▼─────┐
│QnnBackend│ │OnnxBackend │
│(QCM6490) │ │ (default)  │
└─────────┘ └───────────┘
```

- **QNN Backend**: Qualcomm Neural Network SDK for 10 TOPS NPU on Mason Scan 600
- **ONNX Backend**: ONNX Runtime with NNAPI delegate (fallback)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design.

## Quick Start

### Prerequisites

- Android Studio (latest)
- Android NDK r25+
- CMake 3.22+

### Build

```bash
# Clone and setup
git clone <repo>
cd triage-vision-android

# Download models
./scripts/download_models.sh

# Build with Android Studio
# Or via command line:
./gradlew assembleDebug
```

### Models Required

| Model | Size | Purpose |
|-------|------|---------|
| mobileclip_visual.onnx | ~85MB | Fast CLIP classification (MobileCLIP-S1) |
| nursing_text_embeddings.bin | ~56KB | Pre-computed text embeddings for nursing labels |
| yolo11n.ncnn | ~5MB | Motion/object detection |
| SmolVLM-500M-Instruct-Q8_0.gguf | ~437MB | Vision-language model for scene understanding |
| mmproj-SmolVLM-500M-Instruct-Q8_0.gguf | ~109MB | Vision projection for SmolVLM |

Run `python scripts/export_mobileclip.py` to generate CLIP model and embeddings.

## Project Structure

```
triage-vision-android/
├── app/
│   └── src/main/
│       ├── java/com/triage/vision/   # Kotlin/Java code
│       ├── cpp/                       # Native C++ (NCNN, llama.cpp)
│       ├── assets/models/             # ML models
│       └── res/                       # Android resources
├── docs/
│   └── ARCHITECTURE.md
├── scripts/
│   ├── download_models.sh
│   └── export_mobileclip.py          # Export MobileCLIP to ONNX
└── README.md
```

## Configuration

Edit `app/src/main/res/values/config.xml`:

```xml
<resources>
    <!-- Fast pipeline -->
    <integer name="fast_pipeline_fps">15</integer>
    <integer name="stillness_alert_seconds">1800</integer>

    <!-- Slow pipeline -->
    <integer name="vlm_interval_minutes">15</integer>
    <bool name="vlm_trigger_on_stillness">true</bool>

    <!-- Storage -->
    <integer name="retention_hours">24</integer>
</resources>
```

## Privacy & Security

- All inference runs on-device
- No images/video transmitted anywhere
- Observations encrypted at rest (AES-256)
- Automatic data purge after retention period
- No facial recognition or patient identification

## License

Proprietary - All rights reserved
