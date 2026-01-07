#!/bin/bash
# Download required ML models for Triage Vision Android

set -e

MODEL_DIR="app/src/main/assets/models"
mkdir -p "$MODEL_DIR"

echo "=== Downloading models for Triage Vision Android ==="

# SmolVLM-500M GGUF (Q4_K_S quantization)
echo ""
echo "[1/3] Downloading SmolVLM-500M (Q4_K_S)..."
if [ ! -f "$MODEL_DIR/smolvlm-500m-q4_k_s.gguf" ]; then
    curl -L -o "$MODEL_DIR/smolvlm-500m-q4_k_s.gguf" \
        "https://huggingface.co/mradermacher/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct.Q4_K_S.gguf"
    echo "Downloaded SmolVLM-500M"
else
    echo "SmolVLM-500M already exists, skipping"
fi

# SmolVLM mmproj (vision encoder)
echo ""
echo "[2/3] Downloading SmolVLM vision encoder..."
if [ ! -f "$MODEL_DIR/mmproj-smolvlm.gguf" ]; then
    # Note: You may need to convert this from the HF model
    echo "WARNING: mmproj file needs to be converted from HuggingFace model"
    echo "See: https://github.com/ggerganov/llama.cpp/tree/master/examples/llava"
    # Placeholder - actual URL depends on availability
    # curl -L -o "$MODEL_DIR/mmproj-smolvlm.gguf" "URL"
else
    echo "mmproj already exists, skipping"
fi

# YOLO11n NCNN model
echo ""
echo "[3/3] Downloading YOLO11n NCNN model..."
if [ ! -f "$MODEL_DIR/yolo11n.ncnn.bin" ]; then
    echo "YOLO11n NCNN model needs to be exported from Ultralytics"
    echo ""
    echo "Run this Python code to export:"
    echo "  from ultralytics import YOLO"
    echo "  model = YOLO('yolo11n.pt')"
    echo "  model.export(format='ncnn')"
    echo ""
    echo "Then copy yolo11n_ncnn_model/ contents to $MODEL_DIR/"
else
    echo "YOLO11n already exists, skipping"
fi

echo ""
echo "=== Model download complete ==="
echo ""
echo "Model sizes:"
ls -lh "$MODEL_DIR"/ 2>/dev/null || echo "No models downloaded yet"
