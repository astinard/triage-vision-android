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

    // Load model (actual asset paths)
    std::string param_path = model_path + "/yolo11n_ncnn_model/model.ncnn.param";
    std::string bin_path = model_path + "/yolo11n_ncnn_model/model.ncnn.bin";

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

    // Parse YOLO11 output
    // YOLO11 output shape: [8400, 84] where 84 = 4 (bbox) + 80 (class probs)
    // bbox format: x_center, y_center, w, h (in pixels, need to scale to input size)
    // class probs: already sigmoid applied, no separate objectness score
    person_detected_ = false;
    fall_detected_ = false;

    LOGI("YOLO output: w=%d h=%d c=%d", out.w, out.h, out.c);

    // YOLO11 NCNN output is [84, 8400] where:
    // - 84 rows = 4 (bbox: cx, cy, w, h) + 80 (class probs)
    // - 8400 columns = number of detections
    // So out.h = 84 (features), out.w = 8400 (detections)
    const int num_classes = 80;
    const int num_dets = out.w;  // 8400 detections
    const int feat_dim = out.h;  // 84 features per detection

    for (int i = 0; i < num_dets; i++) {
        // Access column i - each detection is a column, not a row
        // row(j) gives feature j, and we want element i from that row
        float cx = out.row(0)[i];
        float cy = out.row(1)[i];
        float bw = out.row(2)[i];
        float bh = out.row(3)[i];

        // Find best class (rows 4-83 are class probs)
        int best_class = 0;
        float best_score = 0;
        for (int j = 4; j < feat_dim && j < 4 + num_classes; j++) {
            float score = out.row(j)[i];
            if (score > best_score) {
                best_score = score;
                best_class = j - 4;
            }
        }

        // In YOLO11, class probability IS the confidence
        float confidence = best_score;
        if (confidence < conf_threshold_) continue;

        // Convert to box coordinates (scale from 640x640 to actual image size)
        Detection det;
        det.x1 = (cx - bw/2) * width / input_width_;
        det.y1 = (cy - bh/2) * height / input_height_;
        det.x2 = (cx + bw/2) * width / input_width_;
        det.y2 = (cy + bh/2) * height / input_height_;
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
