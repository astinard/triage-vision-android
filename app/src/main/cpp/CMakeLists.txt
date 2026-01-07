cmake_minimum_required(VERSION 3.22.1)
project(triage_vision_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Vulkan QUIET)

# ============================================================================
# NCNN Configuration (Fast Pipeline)
# ============================================================================
# Download NCNN or point to local installation
set(NCNN_DIR "${CMAKE_SOURCE_DIR}/ncnn" CACHE PATH "NCNN root directory")

if(EXISTS "${NCNN_DIR}/include/ncnn/net.h")
    message(STATUS "Found NCNN at ${NCNN_DIR}")
    include_directories(${NCNN_DIR}/include)
    link_directories(${NCNN_DIR}/lib)
    set(NCNN_FOUND TRUE)
else()
    message(WARNING "NCNN not found at ${NCNN_DIR}")
    message(STATUS "Download NCNN: https://github.com/Tencent/ncnn/releases")
    set(NCNN_FOUND FALSE)
endif()

# ============================================================================
# llama.cpp Configuration (Slow Pipeline - VLM)
# ============================================================================
set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/llama.cpp" CACHE PATH "llama.cpp root directory")

if(EXISTS "${LLAMA_DIR}/include/llama.h")
    message(STATUS "Found llama.cpp at ${LLAMA_DIR}")
    include_directories(${LLAMA_DIR}/include)
    link_directories(${LLAMA_DIR}/lib)
    set(LLAMA_FOUND TRUE)
else()
    message(WARNING "llama.cpp not found at ${LLAMA_DIR}")
    message(STATUS "Build llama.cpp for Android: https://github.com/ggerganov/llama.cpp/blob/master/docs/android.md")
    set(LLAMA_FOUND FALSE)
endif()

# ============================================================================
# Native Library Sources
# ============================================================================

# Fast Pipeline - YOLO/Motion Detection
set(FAST_PIPELINE_SOURCES
    fast_pipeline/yolo_detector.cpp
    fast_pipeline/motion_analyzer.cpp
    fast_pipeline/pose_estimator.cpp
)

# Depth Processing (always built - used for ToF sensor support)
set(DEPTH_SOURCES
    fast_pipeline/depth_processor.cpp
)

# Slow Pipeline - VLM Inference
set(SLOW_PIPELINE_SOURCES
    slow_pipeline/vlm_inference.cpp
    slow_pipeline/image_processor.cpp
)

# JNI Bridge
set(JNI_SOURCES
    jni/native_bridge.cpp
)

# ============================================================================
# Build Native Library
# ============================================================================
add_library(
    triage_vision
    SHARED
    ${JNI_SOURCES}
    ${DEPTH_SOURCES}
    # Conditionally add pipeline sources based on dependencies
)

# Link libraries
target_link_libraries(
    triage_vision
    android
    log
    jnigraphics
)

# Link NCNN if available
if(NCNN_FOUND)
    target_sources(triage_vision PRIVATE ${FAST_PIPELINE_SOURCES})
    target_link_libraries(triage_vision ncnn)
    target_compile_definitions(triage_vision PRIVATE HAVE_NCNN=1)

    # Link Vulkan for GPU acceleration
    if(Vulkan_FOUND)
        target_link_libraries(triage_vision Vulkan::Vulkan)
        target_compile_definitions(triage_vision PRIVATE HAVE_VULKAN=1)
    endif()
endif()

# Link llama.cpp if available
if(LLAMA_FOUND)
    target_sources(triage_vision PRIVATE ${SLOW_PIPELINE_SOURCES})
    target_link_libraries(triage_vision llama ggml)
    target_compile_definitions(triage_vision PRIVATE HAVE_LLAMA=1)
endif()

# Optimization flags
target_compile_options(triage_vision PRIVATE
    -O3
    -ffast-math
    -fno-rtti
    -fno-exceptions
)
