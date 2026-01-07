#pragma once

#include "yolo_detector.h"
#include <deque>
#include <chrono>

namespace triage {

struct PoseHistory {
    Pose pose;
    int64_t timestamp;
    float confidence;
};

class PoseEstimator {
public:
    PoseEstimator();
    ~PoseEstimator();

    /**
     * Update pose estimate from detections
     * @param detections YOLO detections from current frame
     */
    void update(const std::vector<Detection>& detections);

    /**
     * Get current estimated pose
     */
    Pose getCurrentPose() const { return current_pose_; }

    /**
     * Get pose confidence (0-1)
     */
    float getConfidence() const { return pose_confidence_; }

    /**
     * Check if pose changed recently
     * @param within_seconds Check within this time window
     */
    bool hasPoseChanged(int within_seconds = 60) const;

    /**
     * Get previous pose (before last change)
     */
    Pose getPreviousPose() const { return previous_pose_; }

    /**
     * Get time in current pose (seconds)
     */
    int64_t getTimeInCurrentPose() const;

    /**
     * Reset pose tracking
     */
    void reset();

private:
    Pose current_pose_ = Pose::UNKNOWN;
    Pose previous_pose_ = Pose::UNKNOWN;
    float pose_confidence_ = 0.0f;

    int64_t pose_start_time_ = 0;
    int64_t last_pose_change_time_ = 0;

    std::deque<PoseHistory> pose_history_;
    static const int MAX_HISTORY = 100;

    Pose estimatePoseFromBox(const Detection& person_detection);
    void updatePoseHistory(Pose pose, float confidence);
};

} // namespace triage
