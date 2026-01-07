#include "pose_estimator.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>

#define LOG_TAG "PoseEstimator"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace triage {

PoseEstimator::PoseEstimator() {
    reset();
}

PoseEstimator::~PoseEstimator() {
}

void PoseEstimator::update(const std::vector<Detection>& detections) {
    // Find person detection
    const Detection* person = nullptr;
    float best_conf = 0.0f;

    for (const auto& det : detections) {
        if (det.class_id == 0 && det.confidence > best_conf) { // person class
            person = &det;
            best_conf = det.confidence;
        }
    }

    if (!person) {
        // No person detected - maintain last known pose with reduced confidence
        pose_confidence_ *= 0.95f;
        return;
    }

    // Estimate pose from bounding box
    Pose estimated = estimatePoseFromBox(*person);

    // Smooth pose changes (require multiple consistent frames)
    updatePoseHistory(estimated, best_conf);
}

Pose PoseEstimator::estimatePoseFromBox(const Detection& det) {
    float box_width = det.x2 - det.x1;
    float box_height = det.y2 - det.y1;
    float aspect_ratio = box_width / std::max(box_height, 1.0f);

    // Normalized Y position (0 = top, 1 = bottom)
    float center_y = (det.y1 + det.y2) / 2.0f;

    // Heuristics for pose estimation from bounding box
    //
    // Standing: Tall, narrow box (aspect < 0.5)
    // Sitting: Medium aspect (0.5 - 1.0), typically in lower half
    // Lying: Wide box (aspect > 1.5)
    // Fallen: Very wide box near bottom of frame

    if (aspect_ratio > 2.0f && center_y > 0.7f) {
        return Pose::FALLEN;
    } else if (aspect_ratio > 1.5f) {
        return Pose::LYING;
    } else if (aspect_ratio < 0.5f) {
        return Pose::STANDING;
    } else if (aspect_ratio < 1.0f && center_y > 0.4f) {
        return Pose::SITTING;
    } else if (aspect_ratio < 0.7f) {
        return Pose::STANDING;
    }

    return Pose::UNKNOWN;
}

void PoseEstimator::updatePoseHistory(Pose pose, float confidence) {
    auto now = std::chrono::system_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    // Add to history
    PoseHistory entry{pose, now_ms, confidence};
    pose_history_.push_back(entry);

    // Limit history size
    while (pose_history_.size() > MAX_HISTORY) {
        pose_history_.pop_front();
    }

    // Count recent poses (last 10 frames)
    int pose_counts[5] = {0};
    float pose_conf_sum[5] = {0};
    int recent_count = std::min((int)pose_history_.size(), 10);

    auto it = pose_history_.rbegin();
    for (int i = 0; i < recent_count; i++, it++) {
        int idx = static_cast<int>(it->pose);
        pose_counts[idx]++;
        pose_conf_sum[idx] += it->confidence;
    }

    // Find most common pose
    int best_count = 0;
    Pose best_pose = Pose::UNKNOWN;
    float best_conf = 0.0f;

    for (int i = 0; i < 5; i++) {
        if (pose_counts[i] > best_count) {
            best_count = pose_counts[i];
            best_pose = static_cast<Pose>(i);
            best_conf = pose_conf_sum[i] / pose_counts[i];
        }
    }

    // Update current pose if we have enough confidence
    if (best_count >= 5 || (best_count >= 3 && best_conf > 0.7f)) {
        if (best_pose != current_pose_) {
            previous_pose_ = current_pose_;
            current_pose_ = best_pose;
            pose_start_time_ = now_ms;
            last_pose_change_time_ = now_ms;
            LOGI("Pose changed: %d -> %d", static_cast<int>(previous_pose_),
                 static_cast<int>(current_pose_));
        }
    }

    pose_confidence_ = best_conf;
}

bool PoseEstimator::hasPoseChanged(int within_seconds) const {
    auto now = std::chrono::system_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    int64_t threshold_ms = within_seconds * 1000L;
    return (now_ms - last_pose_change_time_) < threshold_ms;
}

int64_t PoseEstimator::getTimeInCurrentPose() const {
    auto now = std::chrono::system_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    return (now_ms - pose_start_time_) / 1000;
}

void PoseEstimator::reset() {
    current_pose_ = Pose::UNKNOWN;
    previous_pose_ = Pose::UNKNOWN;
    pose_confidence_ = 0.0f;
    pose_history_.clear();

    auto now = std::chrono::system_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    pose_start_time_ = now_ms;
    last_pose_change_time_ = now_ms;
}

} // namespace triage
