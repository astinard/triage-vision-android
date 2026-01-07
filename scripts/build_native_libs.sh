#!/bin/bash
# Build native libraries for Triage Vision Android
# Target: Mason Scan 600 (Qualcomm QCM6490, arm64-v8a)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/native_build"
LIBS_DIR="$PROJECT_DIR/app/src/main/cpp"

# Android NDK path (adjust as needed)
ANDROID_NDK="${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/25.2.9519653}"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-28"

echo "=============================================="
echo "  Triage Vision Native Libraries Builder"
echo "=============================================="
echo ""
echo "Project dir: $PROJECT_DIR"
echo "Build dir: $BUILD_DIR"
echo "NDK: $ANDROID_NDK"
echo "ABI: $ANDROID_ABI"
echo ""

# Check NDK
if [ ! -d "$ANDROID_NDK" ]; then
    echo "ERROR: Android NDK not found at $ANDROID_NDK"
    echo ""
    echo "Please install NDK via Android Studio SDK Manager or set ANDROID_NDK env var"
    echo "Recommended: NDK r25 or later"
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ============================================================================
# Build NCNN
# ============================================================================
build_ncnn() {
    echo ""
    echo "=== Building NCNN ==="
    echo ""

    NCNN_DIR="$BUILD_DIR/ncnn"
    NCNN_BUILD="$BUILD_DIR/ncnn-build"
    NCNN_INSTALL="$LIBS_DIR/ncnn"

    # Clone if not exists
    if [ ! -d "$NCNN_DIR" ]; then
        echo "Cloning NCNN..."
        git clone --depth 1 https://github.com/Tencent/ncnn.git "$NCNN_DIR"
    else
        echo "NCNN already cloned, pulling latest..."
        cd "$NCNN_DIR" && git pull && cd "$BUILD_DIR"
    fi

    # Build
    mkdir -p "$NCNN_BUILD"
    cd "$NCNN_BUILD"

    cmake "$NCNN_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$ANDROID_ABI" \
        -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
        -DANDROID_STL=c++_shared \
        -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_VULKAN=ON \
        -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_TOOLS=OFF \
        -DNCNN_BUILD_BENCHMARK=OFF \
        -DNCNN_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX="$NCNN_INSTALL"

    make -j$(nproc)
    make install

    echo "NCNN installed to: $NCNN_INSTALL"
}

# ============================================================================
# Build llama.cpp
# ============================================================================
build_llama_cpp() {
    echo ""
    echo "=== Building llama.cpp ==="
    echo ""

    LLAMA_DIR="$BUILD_DIR/llama.cpp"
    LLAMA_BUILD="$BUILD_DIR/llama-build"
    LLAMA_INSTALL="$LIBS_DIR/llama.cpp"

    # Clone if not exists
    if [ ! -d "$LLAMA_DIR" ]; then
        echo "Cloning llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    else
        echo "llama.cpp already cloned, pulling latest..."
        cd "$LLAMA_DIR" && git pull && cd "$BUILD_DIR"
    fi

    # Build
    mkdir -p "$LLAMA_BUILD"
    cd "$LLAMA_BUILD"

    cmake "$LLAMA_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$ANDROID_ABI" \
        -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
        -DANDROID_STL=c++_shared \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_OPENMP=OFF \
        -DGGML_OPENCL=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DCMAKE_INSTALL_PREFIX="$LLAMA_INSTALL"

    make -j$(nproc)
    make install

    echo "llama.cpp installed to: $LLAMA_INSTALL"
}

# ============================================================================
# Download pre-built NCNN (alternative to building)
# ============================================================================
download_ncnn_prebuilt() {
    echo ""
    echo "=== Downloading pre-built NCNN ==="
    echo ""

    NCNN_INSTALL="$LIBS_DIR/ncnn"
    NCNN_VERSION="20231027"

    mkdir -p "$BUILD_DIR/downloads"
    cd "$BUILD_DIR/downloads"

    # Download Android release
    if [ ! -f "ncnn-$NCNN_VERSION-android-vulkan.zip" ]; then
        echo "Downloading NCNN $NCNN_VERSION..."
        curl -L -o "ncnn-$NCNN_VERSION-android-vulkan.zip" \
            "https://github.com/Tencent/ncnn/releases/download/$NCNN_VERSION/ncnn-$NCNN_VERSION-android-vulkan.zip"
    fi

    # Extract
    unzip -o "ncnn-$NCNN_VERSION-android-vulkan.zip"

    # Copy arm64-v8a libs
    mkdir -p "$NCNN_INSTALL/lib"
    mkdir -p "$NCNN_INSTALL/include"

    cp -r "ncnn-$NCNN_VERSION-android-vulkan/$ANDROID_ABI/lib/"* "$NCNN_INSTALL/lib/"
    cp -r "ncnn-$NCNN_VERSION-android-vulkan/$ANDROID_ABI/include/"* "$NCNN_INSTALL/include/"

    echo "NCNN pre-built installed to: $NCNN_INSTALL"
}

# ============================================================================
# Main
# ============================================================================

echo "Select build option:"
echo "  1) Build NCNN from source (recommended)"
echo "  2) Download pre-built NCNN"
echo "  3) Build llama.cpp from source"
echo "  4) Build all from source"
echo "  5) Quick setup (pre-built NCNN + build llama.cpp)"
echo ""

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        build_ncnn
        ;;
    2)
        download_ncnn_prebuilt
        ;;
    3)
        build_llama_cpp
        ;;
    4)
        build_ncnn
        build_llama_cpp
        ;;
    5)
        download_ncnn_prebuilt
        build_llama_cpp
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "  Build complete!"
echo "=============================================="
echo ""
echo "Native libraries installed to:"
echo "  $LIBS_DIR/ncnn"
echo "  $LIBS_DIR/llama.cpp"
echo ""
echo "Next steps:"
echo "  1. Download models: ./scripts/download_models.sh"
echo "  2. Open project in Android Studio"
echo "  3. Build and deploy to Mason Scan 600"
