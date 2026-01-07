# Triage Vision Android - Architecture

## Overview

On-device nursing triage vision system for Mason Scan 600. Fully HIPAA-compliant with no data leaving the device.

## Target Hardware

| Spec | Value |
|------|-------|
| Device | Mason Scan 600 |
| Processor | Qualcomm QCM6490 |
| NPU | 10 TOPS (Hexagon DSP) |
| GPU | Adreno 643 (Vulkan) |
| RAM | 6GB |
| Storage | 128GB + SD |
| Camera | 16MP + ToF |
| OS | Android 13 |

## Dual Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMERA INPUT (16MP)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────────┐  ┌─────────────────────────────────────┐
│   FAST PIPELINE     │  │        SLOW PIPELINE                │
│   (Continuous)      │  │        (Interval/Triggered)         │
├─────────────────────┤  ├─────────────────────────────────────┤
│ Framework: NCNN     │  │ Framework: llama.cpp                │
│ Model: YOLO11n      │  │ Model: SmolVLM-500M (Q4_K_S)        │
│ Task: Motion/Pose   │  │ Task: Scene understanding           │
│ FPS: 15-30          │  │ Interval: 5-15 min or on trigger    │
│ Latency: <50ms      │  │ Latency: 2-5 sec                    │
│ Power: ~200mW       │  │ Power: ~2W burst                    │
└─────────┬───────────┘  └──────────────────┬──────────────────┘
          │                                  │
          │ Triggers on:                     │
          │ • No motion > 30 min             │
          │ • Fall detected                  │
          │ • Unusual pose                   │
          ▼                                  │
┌─────────────────────┐                      │
│   ALERT SYSTEM      │                      │
│   • Immediate alert │                      │
│   • Log event       │◄─────────────────────┘
│   • Trigger VLM     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    AUTO-CHARTING ENGINE                      │
├─────────────────────────────────────────────────────────────┤
│ Input: VLM observations + motion history + timestamps        │
│ Output: Structured nursing chart entries                     │
│ Format: HL7 FHIR compatible JSON                            │
└─────────────────────────────────────────────────────────────┘
```

## Fast Pipeline: Motion/Stillness Detection

### Purpose
Continuous monitoring for:
- Patient movement/stillness
- Fall detection
- Pose changes (sitting up, lying down, standing)
- Presence/absence in frame

### Technology Stack
- **Framework**: NCNN (optimized for ARM + Vulkan)
- **Model**: YOLO11n (2.6M params) or MobileNet-SSD
- **Acceleration**: Vulkan GPU (Adreno 643) or Hexagon NPU
- **Input**: 640x480 @ 15-30 FPS

### Detection Classes
```
- person (presence)
- person_lying
- person_sitting
- person_standing
- fall_detected
```

### Motion Analysis
```cpp
// Frame differencing for motion detection
// Optical flow for movement direction
// Stillness timer triggers VLM analysis
```

## Slow Pipeline: Video Language Model

### Purpose
Periodic deep analysis for:
- Detailed scene understanding
- Patient state assessment
- Natural language observations
- Context-aware charting

### Technology Stack
- **Framework**: llama.cpp (Android build)
- **Model**: SmolVLM-500M-Instruct (Q4_K_S, ~293MB)
- **Acceleration**: CPU + potential Vulkan
- **Input**: Single frame or short video clip

### Prompt Template
```
Analyze this patient monitoring image. Describe:
1. Patient position (lying, sitting, standing)
2. Alertness level (awake, sleeping, eyes closed)
3. Any visible medical equipment (IV, monitors, oxygen)
4. Any concerns or notable observations
5. General patient comfort assessment

Respond in structured JSON format.
```

### Output Schema
```json
{
  "timestamp": "ISO8601",
  "observation": {
    "position": "lying_left_lateral | lying_supine | sitting | standing",
    "alertness": "awake | sleeping | drowsy | unresponsive",
    "movement_level": "none | minimal | moderate | active",
    "equipment_visible": ["iv_line", "pulse_oximeter", "nasal_cannula"],
    "concerns": ["none"] | ["description of concern"],
    "comfort_assessment": "comfortable | restless | in_distress"
  },
  "chart_note": "Free text nursing observation summary"
}
```

## Auto-Charting Output

### HL7 FHIR Compatible Format
```json
{
  "resourceType": "Observation",
  "status": "final",
  "category": [{
    "coding": [{
      "system": "http://terminology.hl7.org/CodeSystem/observation-category",
      "code": "survey",
      "display": "Survey"
    }]
  }],
  "code": {
    "coding": [{
      "system": "http://loinc.org",
      "code": "75292-3",
      "display": "Patient monitoring observation"
    }]
  },
  "effectiveDateTime": "2026-01-06T14:32:00Z",
  "valueString": "Patient resting comfortably in left lateral position...",
  "component": [...]
}
```

## Data Flow

1. **Camera** captures continuous video stream
2. **Fast Pipeline** processes every frame for motion/pose
3. **Motion events** logged with timestamps
4. **Slow Pipeline** triggered:
   - Every 15 minutes (configurable)
   - When stillness exceeds threshold
   - When fall detected
   - Manual trigger by nurse
5. **VLM** generates observation JSON
6. **Chart Engine** formats for EMR export
7. **Local storage** keeps 24h rolling buffer (encrypted)

## Privacy & Security

- All processing on-device (no cloud)
- No video/images transmitted
- Only structured observations exportable
- AES-256 encryption for local storage
- Automatic purge after 24h (configurable)
- No facial recognition or patient identification
- Barcode scan links observations to patient record

## Model Files Required

| Model | Size | Location |
|-------|------|----------|
| yolo11n.ncnn.bin | ~5MB | assets/models/ |
| yolo11n.ncnn.param | ~50KB | assets/models/ |
| smolvlm-500m-q4_k_s.gguf | ~293MB | assets/models/ |
| mmproj-smolvlm.gguf | ~100MB | assets/models/ |

## Performance Targets

| Metric | Target |
|--------|--------|
| Fast pipeline FPS | 15-30 |
| Fast pipeline latency | <50ms |
| VLM inference time | <5 sec |
| Battery life (monitoring) | 8+ hours |
| Storage per day | <100MB |
