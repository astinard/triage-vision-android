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

    g_initialized = false;
    LOGI("Native cleanup complete");
}

} // extern "C"
