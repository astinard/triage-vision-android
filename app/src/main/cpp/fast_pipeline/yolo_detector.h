#pragma once

#include <vector>
#include <string>

#ifdef HAVE_NCNN
#include <ncnn/net.h>
#endif

namespace triage {

struct Detection {
    float x1, y1, x2, y2;  // Bounding box
    float confidence;
    int class_id;
    std::string class_name;
};

struct PoseKeypoint {
    float x, y;
    float confidence;
};

enum class Pose {
    UNKNOWN = 0,
    LYING = 1,
    SITTING = 2,
    STANDING = 3,
    FALLEN = 4
};

class YoloDetector {
public:
    YoloDetector();
    ~YoloDetector();

    /**
     * Initialize the detector with model files
     * @param model_path Path to directory containing .param and .bin files
     * @param use_gpu Whether to use Vulkan GPU acceleration
     * @return true on success
     */
    bool init(const std::string& model_path, bool use_gpu = true);

    /**
     * Run detection on an image
     * @param pixels RGBA pixel data
     * @param width Image width
     * @param height Image height
     * @return Vector of detections
     */
    std::vector<Detection> detect(const uint8_t* pixels, int width, int height);

    /**
     * Check if person is detected in frame
     */
    bool isPersonDetected() const { return person_detected_; }

    /**
     * Get estimated pose from last detection
     */
    Pose getEstimatedPose() const { return estimated_pose_; }

    /**
     * Check if fall is detected
     */
    bool isFallDetected() const { return fall_detected_; }

    /**
     * Cleanup resources
     */
    void cleanup();

private:
    bool initialized_ = false;
    bool person_detected_ = false;
    bool fall_detected_ = false;
    Pose estimated_pose_ = Pose::UNKNOWN;

#ifdef HAVE_NCNN
    ncnn::Net net_;
    ncnn::Option opt_;
#endif

    // Detection parameters
    float conf_threshold_ = 0.5f;
    float nms_threshold_ = 0.45f;
    int input_width_ = 640;
    int input_height_ = 640;

    // Class names for YOLO
    std::vector<std::string> class_names_;

    void estimatePose(const std::vector<Detection>& detections);
    bool checkForFall(const std::vector<Detection>& detections);
};

} // namespace triage
