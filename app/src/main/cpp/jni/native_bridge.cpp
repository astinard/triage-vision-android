/**
 * JNI Bridge for Triage Vision Native Libraries
 *
 * Provides Java/Kotlin interface to:
 * - Fast Pipeline (NCNN/YOLO for motion detection)
 * - Slow Pipeline (llama.cpp/SmolVLM for scene understanding)
 */

#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <string>
#include <vector>
#include <memory>

#define LOG_TAG "TriageVisionNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#ifdef HAVE_NCNN
#include "../fast_pipeline/yolo_detector.h"
#include "../fast_pipeline/motion_analyzer.h"
#include "../fast_pipeline/pose_estimator.h"
#endif

#include "../fast_pipeline/depth_processor.h"

#ifdef HAVE_LLAMA
#include "../slow_pipeline/vlm_inference.h"
#endif

#include "../slow_pipeline/image_processor.h"

// Global instances
#ifdef HAVE_NCNN
static std::unique_ptr<triage::YoloDetector> g_yolo_detector;
static std::unique_ptr<triage::MotionAnalyzer> g_motion_analyzer;
static std::unique_ptr<triage::PoseEstimator> g_pose_estimator;
#endif

#ifdef HAVE_LLAMA
static std::unique_ptr<triage::VLMInference> g_vlm;
#endif

// Depth processor (always available)
static std::unique_ptr<triage::DepthProcessor> g_depth_processor;

static std::string g_model_path;
static bool g_initialized = false;

extern "C" {

// ============================================================================
// Initialization
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_triage_vision_native_NativeBridge_init(
    JNIEnv *env,
    jobject thiz,
    jstring model_path
) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    g_model_path = path;
    LOGI("Initializing Triage Vision Native with model path: %s", path);

    int result = 0;

#ifdef HAVE_NCNN
    LOGI("NCNN support enabled - initializing fast pipeline");

    // Initialize YOLO detector
    g_yolo_detector = std::make_unique<triage::YoloDetector>();
    if (!g_yolo_detector->init(g_model_path, true)) {
        LOGE("Failed to initialize YOLO detector");
        result = -1;
    }

    // Initialize motion analyzer
    g_motion_analyzer = std::make_unique<triage::MotionAnalyzer>();
    g_motion_analyzer->init(0.05f, 30);

    // Initialize pose estimator
    g_pose_estimator = std::make_unique<triage::PoseEstimator>();

#else
    LOGI("NCNN support not available - fast pipeline disabled");
#endif

#ifdef HAVE_LLAMA
    LOGI("llama.cpp support enabled - initializing slow pipeline");

    std::string vlm_path = g_model_path + "/smolvlm-500m-q4_k_s.gguf";
    std::string mmproj_path = g_model_path + "/mmproj-smolvlm.gguf";

    g_vlm = std::make_unique<triage::VLMInference>();
    if (!g_vlm->init(vlm_path, mmproj_path, 4, 0)) {
        LOGE("Failed to initialize VLM");
        // Don't fail completely - VLM is optional
    }

#else
    LOGI("llama.cpp support not available - slow pipeline disabled");
#endif

    g_initialized = (result == 0);
    env->ReleaseStringUTFChars(model_path, path);
    return result;
}

// ============================================================================
// Fast Pipeline - Motion/Pose Detection
// ============================================================================

JNIEXPORT jstring JNICALL
Java_com_triage_vision_native_NativeBridge_detectMotion(
    JNIEnv *env,
    jobject thiz,
    jobject bitmap
) {
    // Get bitmap info
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to get bitmap info");
        return env->NewStringUTF("{}");
    }

    // Lock pixels
    void *pixels;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to lock bitmap pixels");
        return env->NewStringUTF("{}");
    }

    std::string result_json = "{}";

#ifdef HAVE_NCNN
    if (g_yolo_detector && g_motion_analyzer && g_pose_estimator) {
        // Run YOLO detection
        auto detections = g_yolo_detector->detect(
            static_cast<uint8_t*>(pixels), info.width, info.height);

        // Analyze motion
        auto motion_state = g_motion_analyzer->analyze(
            static_cast<uint8_t*>(pixels), info.width, info.height);

        // Update pose estimator
        g_pose_estimator->update(detections);

        // Build JSON result
        char json_buf[1024];
        snprintf(json_buf, sizeof(json_buf),
            R"({"person_detected": %s, "pose": %d, "motion_level": %.3f, )"
            R"("fall_detected": %s, "seconds_since_motion": %lld, "detection_count": %zu})",
            g_yolo_detector->isPersonDetected() ? "true" : "false",
            static_cast<int>(g_pose_estimator->getCurrentPose()),
            motion_state.motion_level,
            g_yolo_detector->isFallDetected() ? "true" : "false",
            (long long)g_motion_analyzer->getSecondsSinceMotion(),
            detections.size()
        );
        result_json = json_buf;
    }
#endif

    // Unlock pixels
    AndroidBitmap_unlockPixels(env, bitmap);

    return env->NewStringUTF(result_json.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_triage_vision_native_NativeBridge_isPersonDetected(
    JNIEnv *env,
    jobject thiz,
    jobject bitmap
) {
#ifdef HAVE_NCNN
    if (g_yolo_detector) {
        return g_yolo_detector->isPersonDetected() ? JNI_TRUE : JNI_FALSE;
    }
#endif
    return JNI_FALSE;
}

JNIEXPORT jfloat JNICALL
Java_com_triage_vision_native_NativeBridge_getMotionLevel(
    JNIEnv *env,
    jobject thiz
) {
#ifdef HAVE_NCNN
    if (g_motion_analyzer) {
        return g_motion_analyzer->getMotionLevel();
    }
#endif
    return 0.0f;
}

// ============================================================================
// Depth-Enhanced Detection
// ============================================================================

JNIEXPORT jstring JNICALL
Java_com_triage_vision_native_NativeBridge_detectMotionWithDepth(
    JNIEnv *env,
    jobject thiz,
    jobject bitmap,
    jshortArray depth_data,
    jint depth_width,
    jint depth_height
) {
    // Get RGB bitmap info and pixels
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to get bitmap info");
        return env->NewStringUTF(R"({"error": "Failed to get bitmap info"})");
    }

    void *pixels;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to lock bitmap pixels");
        return env->NewStringUTF(R"({"error": "Failed to lock bitmap pixels"})");
    }

    // Get depth array
    jshort* depth_ptr = nullptr;
    if (depth_data != nullptr) {
        depth_ptr = env->GetShortArrayElements(depth_data, nullptr);
    }

    // Initialize depth processor if needed
    if (!g_depth_processor) {
        g_depth_processor = std::make_unique<triage::DepthProcessor>();
    }

    // Update depth map if available
    if (depth_ptr != nullptr && depth_width > 0 && depth_height > 0) {
        g_depth_processor->updateDepthMap(
            reinterpret_cast<uint16_t*>(depth_ptr),
            depth_width,
            depth_height
        );
    }

    std::string result_json = "{}";

#ifdef HAVE_NCNN
    if (g_yolo_detector && g_motion_analyzer && g_pose_estimator) {
        // Run YOLO detection on RGB
        auto detections = g_yolo_detector->detect(
            static_cast<uint8_t*>(pixels), info.width, info.height);

        // Analyze RGB motion
        auto motion_state = g_motion_analyzer->analyze(
            static_cast<uint8_t*>(pixels), info.width, info.height);

        // Update pose estimator
        g_pose_estimator->update(detections);

        // Depth-enhanced analysis
        float distance_meters = 0.0f;
        float depth_motion_level = 0.0f;
        bool depth_fall = false;
        float vertical_drop = 0.0f;
        float fall_confidence = 0.0f;
        float bed_proximity = 0.0f;
        bool in_bed_zone = false;
        float pos_x = 0.0f, pos_y = 0.0f, pos_z = 0.0f;

        if (g_depth_processor->hasDepthData() && !detections.empty()) {
            // Get person bounding box (use first detection)
            auto& det = detections[0];
            triage::BoundingBox person_bbox = {
                det.x1 / static_cast<float>(info.width),
                det.y1 / static_cast<float>(info.height),
                (det.x2 - det.x1) / static_cast<float>(info.width),
                (det.y2 - det.y1) / static_cast<float>(info.height)
            };

            // Fall detection with depth
            auto fall_result = g_depth_processor->detectFall(
                person_bbox, info.width, info.height);
            depth_fall = fall_result.fall_detected;
            vertical_drop = fall_result.vertical_drop_meters;
            fall_confidence = fall_result.confidence;

            // Motion analysis with depth
            auto motion_result = g_depth_processor->analyzeMotion(
                person_bbox, info.width, info.height);
            distance_meters = motion_result.distance_meters;
            depth_motion_level = motion_result.depth_motion_level;
            bed_proximity = motion_result.bed_proximity_meters;
            in_bed_zone = motion_result.in_bed_zone;
            pos_x = motion_result.position_3d.x;
            pos_y = motion_result.position_3d.y;
            pos_z = motion_result.position_3d.z;
        }

        // Combined fall detection (2D + depth)
        bool combined_fall = g_yolo_detector->isFallDetected() || depth_fall;

        // Build JSON result with depth metrics
        char json_buf[2048];
        snprintf(json_buf, sizeof(json_buf),
            R"({)"
            R"("person_detected": %s, )"
            R"("pose": %d, )"
            R"("motion_level": %.3f, )"
            R"("fall_detected": %s, )"
            R"("depth_fall": %s, )"
            R"("vertical_drop_meters": %.3f, )"
            R"("fall_confidence": %.2f, )"
            R"("seconds_since_motion": %lld, )"
            R"("detection_count": %zu, )"
            R"("distance_meters": %.2f, )"
            R"("depth_motion_level": %.3f, )"
            R"("bed_proximity_meters": %.2f, )"
            R"("in_bed_zone": %s, )"
            R"("position_3d": {"x": %.3f, "y": %.3f, "z": %.3f}, )"
            R"("depth_available": %s)"
            R"(})",
            g_yolo_detector->isPersonDetected() ? "true" : "false",
            static_cast<int>(g_pose_estimator->getCurrentPose()),
            motion_state.motion_level,
            combined_fall ? "true" : "false",
            depth_fall ? "true" : "false",
            vertical_drop,
            fall_confidence,
            (long long)g_motion_analyzer->getSecondsSinceMotion(),
            detections.size(),
            distance_meters,
            depth_motion_level,
            bed_proximity,
            in_bed_zone ? "true" : "false",
            pos_x, pos_y, pos_z,
            g_depth_processor->hasDepthData() ? "true" : "false"
        );
        result_json = json_buf;
    }
#endif

    // Cleanup
    if (depth_ptr != nullptr) {
        env->ReleaseShortArrayElements(depth_data, depth_ptr, JNI_ABORT);
    }
    AndroidBitmap_unlockPixels(env, bitmap);

    return env->NewStringUTF(result_json.c_str());
}

JNIEXPORT jfloat JNICALL
Java_com_triage_vision_native_NativeBridge_getDepthAt(
    JNIEnv *env,
    jobject thiz,
    jint x,
    jint y
) {
    if (g_depth_processor) {
        return g_depth_processor->getDepthAt(x, y);
    }
    return -1.0f;
}

JNIEXPORT jfloat JNICALL
Java_com_triage_vision_native_NativeBridge_getAverageDistance(
    JNIEnv *env,
    jobject thiz
) {
    if (g_depth_processor) {
        return g_depth_processor->getAverageDistance();
    }
    return 0.0f;
}

// ============================================================================
// Slow Pipeline - VLM Scene Understanding
// ============================================================================

JNIEXPORT jstring JNICALL
Java_com_triage_vision_native_NativeBridge_analyzeScene(
    JNIEnv *env,
    jobject thiz,
    jobject bitmap,
    jstring prompt
) {
    const char *prompt_str = env->GetStringUTFChars(prompt, nullptr);
    LOGI("VLM analysis with prompt length: %zu", strlen(prompt_str));

    std::string result;

    // Get bitmap data
    AndroidBitmapInfo info;
    void *pixels;

    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to access bitmap");
        env->ReleaseStringUTFChars(prompt, prompt_str);
        return env->NewStringUTF(R"({"error": "Failed to access bitmap"})");
    }

#ifdef HAVE_LLAMA
    if (g_vlm && g_vlm->isInitialized()) {
        auto observation = g_vlm->analyze(
            static_cast<uint8_t*>(pixels), info.width, info.height, prompt_str);

        // Build JSON result
        char json_buf[4096];
        snprintf(json_buf, sizeof(json_buf),
            R"({"success": %s, "position": "%s", "alertness": "%s", )"
            R"("movement_level": "%s", "comfort_assessment": "%s", )"
            R"("chart_note": "%s", "error": "%s"})",
            observation.success ? "true" : "false",
            observation.position.c_str(),
            observation.alertness.c_str(),
            observation.movement_level.c_str(),
            observation.comfort_assessment.c_str(),
            observation.chart_note.c_str(),
            observation.error.c_str()
        );
        result = json_buf;
    } else {
        LOGI("VLM not available, returning placeholder");
        result = R"({
            "success": true,
            "position": "unknown",
            "alertness": "unknown",
            "movement_level": "unknown",
            "equipment_visible": [],
            "concerns": ["VLM not initialized"],
            "comfort_assessment": "unknown",
            "chart_note": "VLM inference not available - placeholder observation"
        })";
    }
#else
    LOGI("llama.cpp not available, returning placeholder");
    result = R"({
        "success": true,
        "position": "unknown",
        "alertness": "unknown",
        "movement_level": "unknown",
        "equipment_visible": [],
        "concerns": ["llama.cpp not compiled"],
        "comfort_assessment": "unknown",
        "chart_note": "VLM inference not available - placeholder observation"
    })";
#endif

    AndroidBitmap_unlockPixels(env, bitmap);
    env->ReleaseStringUTFChars(prompt, prompt_str);

    return env->NewStringUTF(result.c_str());
}

// ============================================================================
// Cleanup
// ============================================================================

JNIEXPORT void JNICALL
Java_com_triage_vision_native_NativeBridge_cleanup(
    JNIEnv *env,
    jobject thiz
) {
    LOGI("Cleaning up native resources");

#ifdef HAVE_NCNN
    if (g_yolo_detector) {
        g_yolo_detector->cleanup();
        g_yolo_detector.reset();
    }
    g_motion_analyzer.reset();
    g_pose_estimator.reset();
#endif

#ifdef HAVE_LLAMA
    if (g_vlm) {
        g_vlm->cleanup();
        g_vlm.reset();
    }
#endif

    // Cleanup depth processor
    if (g_depth_processor) {
        g_depth_processor->reset();
        g_depth_processor.reset();
    }

    g_initialized = false;
    LOGI("Native cleanup complete");
}

} // extern "C"
