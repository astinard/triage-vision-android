#include "depth_processor.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>

#define LOG_TAG "DepthProcessor"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace triage {

DepthProcessor::DepthProcessor() = default;

DepthProcessor::~DepthProcessor() = default;

void DepthProcessor::init(int width, int height) {
    width_ = width;
    height_ = height;
    principal_x_ = width / 2.0f;
    principal_y_ = height / 2.0f;

    // Pre-allocate depth map
    depth_map_.resize(width * height);

    initialized_ = true;
    LOGI("DepthProcessor initialized: %dx%d", width, height);
}

void DepthProcessor::updateDepthMap(const uint16_t* depth_data, int width, int height) {
    if (!initialized_) {
        init(width, height);
    }

    if (width != width_ || height != height_) {
        LOGE("Depth frame size mismatch: expected %dx%d, got %dx%d",
             width_, height_, width, height);
        return;
    }

    // Copy depth data
    std::copy(depth_data, depth_data + (width * height), depth_map_.begin());
}

float DepthProcessor::getDepthAt(int x, int y) const {
    if (!initialized_ || x < 0 || x >= width_ || y < 0 || y >= height_) {
        return -1.0f;
    }

    int idx = y * width_ + x;
    uint16_t raw_depth = depth_map_[idx];

    // Invalid depth values (0 or max)
    if (raw_depth == 0 || raw_depth == 0xFFFF) {
        return -1.0f;
    }

    // DEPTH16 values are in millimeters
    return raw_depth / 1000.0f;
}

float DepthProcessor::getDepthAtNormalized(float norm_x, float norm_y) const {
    int x = static_cast<int>(norm_x * width_);
    int y = static_cast<int>(norm_y * height_);
    return getDepthAt(x, y);
}

DepthStats DepthProcessor::calculateStats(const BoundingBox& bbox) const {
    DepthStats stats = {0, 0, 0, 0, 0, 0};

    if (!initialized_ || depth_map_.empty()) {
        return stats;
    }

    // Convert normalized bbox to pixel coordinates
    int x1 = static_cast<int>(bbox.x * width_);
    int y1 = static_cast<int>(bbox.y * height_);
    int x2 = static_cast<int>((bbox.x + bbox.width) * width_);
    int y2 = static_cast<int>((bbox.y + bbox.height) * height_);

    // Clamp to frame bounds
    x1 = std::max(0, std::min(x1, width_ - 1));
    y1 = std::max(0, std::min(y1, height_ - 1));
    x2 = std::max(0, std::min(x2, width_ - 1));
    y2 = std::max(0, std::min(y2, height_ - 1));

    std::vector<float> valid_depths;

    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            float depth = getDepthAt(x, y);
            if (depth > 0) {
                valid_depths.push_back(depth);
            }
            stats.total_pixels++;
        }
    }

    stats.valid_pixels = static_cast<int>(valid_depths.size());

    if (valid_depths.empty()) {
        return stats;
    }

    // Calculate statistics
    stats.min_meters = *std::min_element(valid_depths.begin(), valid_depths.end());
    stats.max_meters = *std::max_element(valid_depths.begin(), valid_depths.end());

    float sum = std::accumulate(valid_depths.begin(), valid_depths.end(), 0.0f);
    stats.mean_meters = sum / valid_depths.size();

    // Median
    size_t mid = valid_depths.size() / 2;
    std::nth_element(valid_depths.begin(), valid_depths.begin() + mid, valid_depths.end());
    stats.median_meters = valid_depths[mid];

    return stats;
}

Position3D DepthProcessor::estimate3DPosition(
    const BoundingBox& person_bbox,
    int rgb_width,
    int rgb_height
) const {
    Position3D pos = {0, 0, 0};

    if (!initialized_ || depth_map_.empty()) {
        return pos;
    }

    // Calculate center of person bbox in RGB coordinates
    float rgb_cx = (person_bbox.x + person_bbox.width / 2) * rgb_width;
    float rgb_cy = (person_bbox.y + person_bbox.height / 2) * rgb_height;

    // Scale RGB coordinates to depth coordinates
    float scale_x = static_cast<float>(width_) / rgb_width;
    float scale_y = static_cast<float>(height_) / rgb_height;

    int depth_cx = static_cast<int>(rgb_cx * scale_x);
    int depth_cy = static_cast<int>(rgb_cy * scale_y);

    // Get depth at center (use median for robustness)
    int roi_size = 5;  // 5x5 region around center
    float depth = medianDepthInRegion(
        depth_cx - roi_size, depth_cy - roi_size,
        depth_cx + roi_size, depth_cy + roi_size
    );

    if (depth <= 0) {
        // Fallback: try full bbox
        DepthStats stats = calculateStats(person_bbox);
        depth = stats.median_meters;
    }

    if (depth <= 0) {
        return pos;  // No valid depth
    }

    // Convert pixel coords to 3D using pinhole camera model
    // x = (u - cx) * z / fx
    // y = (v - cy) * z / fy
    float u = depth_cx;
    float v = depth_cy;

    pos.x = (u - principal_x_) * depth / focal_length_x_;
    pos.y = (v - principal_y_) * depth / focal_length_y_;
    pos.z = depth;

    return pos;
}

DepthFallResult DepthProcessor::detectFall(
    const BoundingBox& person_bbox,
    int rgb_width,
    int rgb_height
) {
    DepthFallResult result = {false, 0, 0, 0, 0};

    if (!initialized_ || depth_map_.empty()) {
        return result;
    }

    // Get current 3D position
    Position3D current_pos = estimate3DPosition(person_bbox, rgb_width, rgb_height);

    if (current_pos.z <= 0) {
        return result;  // No valid depth
    }

    // Update position history
    updatePositionHistory(current_pos);

    // Current height (Y position, inverted: lower Y = higher up)
    result.current_height_meters = -current_pos.y;  // Assuming camera is looking down

    // Calculate vertical drop and velocity
    result.vertical_drop_meters = calculateVerticalDrop();
    result.drop_velocity_ms = calculateDropVelocity();

    // Fall detection logic
    bool rapid_drop = result.vertical_drop_meters > fall_drop_threshold_;
    bool high_velocity = result.drop_velocity_ms > fall_velocity_threshold_;

    if (rapid_drop && high_velocity) {
        result.fall_detected = true;
        result.confidence = 0.9f;
        LOGI("FALL DETECTED: drop=%.2fm, velocity=%.2fm/s",
             result.vertical_drop_meters, result.drop_velocity_ms);
    } else if (rapid_drop) {
        // Just drop, low velocity (might be sitting down)
        result.fall_detected = false;
        result.confidence = 0.3f;
    } else {
        result.fall_detected = false;
        result.confidence = 0.0f;
    }

    // Store last position
    last_position_ = current_pos;
    last_distance_ = current_pos.z;

    return result;
}

DepthMotionResult DepthProcessor::analyzeMotion(
    const BoundingBox& person_bbox,
    int rgb_width,
    int rgb_height
) {
    DepthMotionResult result = {0, {0, 0, 0}, 0, false, 0};

    if (!initialized_ || depth_map_.empty()) {
        return result;
    }

    // Get current 3D position
    Position3D current_pos = estimate3DPosition(person_bbox, rgb_width, rgb_height);

    result.position_3d = current_pos;
    result.distance_meters = current_pos.z;

    // Calculate Z-axis motion (depth change)
    if (last_position_.z > 0 && current_pos.z > 0) {
        float z_change = std::abs(current_pos.z - last_position_.z);
        result.depth_motion_level = std::min(1.0f, z_change * 10.0f);  // Scale to 0-1
    }

    // Check bed proximity
    float dx = current_pos.x - bed_center_.x;
    float dy = current_pos.y - bed_center_.y;
    float dz = current_pos.z - bed_center_.z;
    result.bed_proximity_meters = std::sqrt(dx*dx + dy*dy + dz*dz);
    result.in_bed_zone = (result.bed_proximity_meters <= bed_radius_);

    // Update state
    last_position_ = current_pos;
    last_distance_ = current_pos.z;

    return result;
}

void DepthProcessor::setBedRegion(const Position3D& center, float radius_meters) {
    bed_center_ = center;
    bed_radius_ = radius_meters;
    LOGI("Bed region set: center=(%.2f, %.2f, %.2f), radius=%.2fm",
         center.x, center.y, center.z, radius_meters);
}

void DepthProcessor::reset() {
    position_history_.clear();
    last_position_ = {0, 0, 0};
    last_distance_ = 0;
    LOGI("DepthProcessor state reset");
}

float DepthProcessor::medianDepthInRegion(int x1, int y1, int x2, int y2) const {
    // Clamp to bounds
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(width_ - 1, x2);
    y2 = std::min(height_ - 1, y2);

    std::vector<float> depths;
    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            float d = getDepthAt(x, y);
            if (d > 0) {
                depths.push_back(d);
            }
        }
    }

    if (depths.empty()) {
        return -1.0f;
    }

    size_t mid = depths.size() / 2;
    std::nth_element(depths.begin(), depths.begin() + mid, depths.end());
    return depths[mid];
}

void DepthProcessor::updatePositionHistory(const Position3D& pos) {
    int64_t now = getCurrentTimeMs();

    position_history_.push_back({pos, now});

    // Remove old entries (keep only last second)
    while (!position_history_.empty() &&
           (now - position_history_.front().timestamp_ms) > fall_time_window_ms_) {
        position_history_.pop_front();
    }

    // Limit size
    while (position_history_.size() > MAX_HISTORY_SIZE) {
        position_history_.pop_front();
    }
}

float DepthProcessor::calculateVerticalDrop() const {
    if (position_history_.size() < 2) {
        return 0.0f;
    }

    // Find max Y (highest point) and current Y
    float max_y = position_history_.front().position.y;
    for (const auto& sample : position_history_) {
        max_y = std::min(max_y, sample.position.y);  // Lower Y = higher up
    }

    float current_y = position_history_.back().position.y;

    // Drop = how much Y has increased (person moved down)
    return current_y - max_y;
}

float DepthProcessor::calculateDropVelocity() const {
    if (position_history_.size() < 2) {
        return 0.0f;
    }

    const auto& first = position_history_.front();
    const auto& last = position_history_.back();

    int64_t time_delta_ms = last.timestamp_ms - first.timestamp_ms;
    if (time_delta_ms <= 0) {
        return 0.0f;
    }

    float y_delta = last.position.y - first.position.y;  // Positive = moved down
    float time_seconds = time_delta_ms / 1000.0f;

    return y_delta / time_seconds;
}

int64_t DepthProcessor::getCurrentTimeMs() const {
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    );
    return ms.count();
}

} // namespace triage
