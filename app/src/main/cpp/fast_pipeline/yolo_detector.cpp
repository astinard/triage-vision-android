#include "yolo_detector.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>

#define LOG_TAG "YoloDetector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace triage {

// COCO class names (subset relevant for patient monitoring)
static const char* COCO_CLASSES[] = {
    "person", "bed", "chair", "couch", "tv", "laptop", "remote",
    "cell phone", "book", "clock", "vase", "bottle", "cup"
};
static const int NUM_COCO_CLASSES = 80;

YoloDetector::YoloDetector() {
    class_names_.assign(COCO_CLASSES, COCO_CLASSES + sizeof(COCO_CLASSES)/sizeof(COCO_CLASSES[0]));
}

YoloDetector::~YoloDetector() {
    cleanup();
}

bool YoloDetector::init(const std::string& model_path, bool use_gpu) {
#ifdef HAVE_NCNN
    LOGI("Initializing YOLO detector from: %s", model_path.c_str());

    // Configure options
    opt_.lightmode = true;
    opt_.num_threads = 4;

    if (use_gpu) {
#ifdef HAVE_VULKAN
        opt_.use_vulkan_compute = ncnn::get_gpu_count() > 0;
        if (opt_.use_vulkan_compute) {
            LOGI("Vulkan GPU acceleration enabled");
        }
#endif
    }

    net_.opt = opt_;

    // Load model
    std::string param_path = model_path + "/yolo11n.ncnn.param";
    std::string bin_path = model_path + "/yolo11n.ncnn.bin";

    int ret = net_.load_param(param_path.c_str());
    if (ret != 0) {
        LOGE("Failed to load param file: %s", param_path.c_str());
        return false;
    }

    ret = net_.load_model(bin_path.c_str());
    if (ret != 0) {
        LOGE("Failed to load model file: %s", bin_path.c_str());
        return false;
    }

    initialized_ = true;
    LOGI("YOLO detector initialized successfully");
    return true;
#else
    LOGE("NCNN not available - detector disabled");
    return false;
#endif
}

std::vector<Detection> YoloDetector::detect(const uint8_t* pixels, int width, int height) {
    std::vector<Detection> detections;

#ifdef HAVE_NCNN
    if (!initialized_) {
        LOGE("Detector not initialized");
        return detections;
    }

    // Create input from pixels
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        pixels, ncnn::Mat::PIXEL_RGBA2RGB,
        width, height,
        input_width_, input_height_
    );

    // Normalize (YOLO expects 0-1)
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // Run inference
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);

    // Parse YOLO output
    // Output shape: [num_detections, 85] (x, y, w, h, conf, 80 class probs)
    person_detected_ = false;
    fall_detected_ = false;

    for (int i = 0; i < out.h; i++) {
        const float* row = out.row(i);

        float obj_conf = row[4];
        if (obj_conf < conf_threshold_) continue;

        // Find best class
        int best_class = 0;
        float best_score = 0;
        for (int j = 5; j < 85; j++) {
            if (row[j] > best_score) {
                best_score = row[j];
                best_class = j - 5;
            }
        }

        float confidence = obj_conf * best_score;
        if (confidence < conf_threshold_) continue;

        // Convert to box coordinates
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        Detection det;
        det.x1 = (cx - w/2) * width / input_width_;
        det.y1 = (cy - h/2) * height / input_height_;
        det.x2 = (cx + w/2) * width / input_width_;
        det.y2 = (cy + h/2) * height / input_height_;
        det.confidence = confidence;
        det.class_id = best_class;
        det.class_name = (best_class < class_names_.size()) ? class_names_[best_class] : "unknown";

        detections.push_back(det);

        // Track person detection
        if (best_class == 0) { // person class
            person_detected_ = true;
        }
    }

    // NMS
    // (simplified - production should use proper NMS)

    // Estimate pose from detections
    estimatePose(detections);

    // Check for fall
    fall_detected_ = checkForFall(detections);

#endif
    return detections;
}

void YoloDetector::estimatePose(const std::vector<Detection>& detections) {
    estimated_pose_ = Pose::UNKNOWN;

    for (const auto& det : detections) {
        if (det.class_id != 0) continue; // Only person class

        // Simple heuristic based on bounding box aspect ratio
        float box_width = det.x2 - det.x1;
        float box_height = det.y2 - det.y1;
        float aspect_ratio = box_width / std::max(box_height, 1.0f);

        // Vertical box (height > width) = standing/sitting
        // Horizontal box (width > height) = lying
        if (aspect_ratio > 1.5f) {
            estimated_pose_ = Pose::LYING;
        } else if (aspect_ratio < 0.4f) {
            estimated_pose_ = Pose::STANDING;
        } else if (det.y1 > 0.5f) { // Lower in frame
            estimated_pose_ = Pose::SITTING;
        } else {
            estimated_pose_ = Pose::STANDING;
        }
        break; // Use first person detected
    }
}

bool YoloDetector::checkForFall(const std::vector<Detection>& detections) {
    // Simple fall detection heuristic
    // More sophisticated version would track pose changes over time

    for (const auto& det : detections) {
        if (det.class_id != 0) continue;

        float box_width = det.x2 - det.x1;
        float box_height = det.y2 - det.y1;
        float aspect_ratio = box_width / std::max(box_height, 1.0f);

        // Very horizontal person near bottom of frame = potential fall
        if (aspect_ratio > 2.0f && det.y2 > 0.8f) {
            estimated_pose_ = Pose::FALLEN;
            return true;
        }
    }

    return false;
}

void YoloDetector::cleanup() {
#ifdef HAVE_NCNN
    if (initialized_) {
        net_.clear();
        initialized_ = false;
        LOGI("YOLO detector cleaned up");
    }
#endif
}

} // namespace triage
