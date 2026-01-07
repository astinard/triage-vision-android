#pragma once

#include <vector>
#include <cstdint>
#include <deque>

namespace triage {

/**
 * 3D position in meters relative to camera
 */
struct Position3D {
    float x;  // Horizontal (right is positive)
    float y;  // Vertical (down is positive)
    float z;  // Depth (away from camera is positive)
};

/**
 * Bounding box in pixel coordinates (normalized 0-1)
 */
struct BoundingBox {
    float x, y;       // Top-left corner (normalized)
    float width, height;  // Size (normalized)
};

/**
 * Depth statistics for a region
 */
struct DepthStats {
    float min_meters;
    float max_meters;
    float mean_meters;
    float median_meters;
    int valid_pixels;
    int total_pixels;
};

/**
 * Result of depth-enhanced fall detection
 */
struct DepthFallResult {
    bool fall_detected;
    float vertical_drop_meters;   // How far the person dropped
    float drop_velocity_ms;       // Speed of descent (m/s)
    float current_height_meters;  // Current height above floor
    float confidence;             // 0.0-1.0
};

/**
 * Depth-enhanced motion analysis result
 */
struct DepthMotionResult {
    float distance_meters;        // Distance from camera to person
    Position3D position_3d;       // 3D centroid position
    float depth_motion_level;     // Motion in Z-axis (0-1)
    bool in_bed_zone;             // Within configured bed region
    float bed_proximity_meters;   // Distance from bed center
};

/**
 * Processes depth sensor data for patient monitoring.
 *
 * Provides:
 * - Fall detection via vertical drop analysis
 * - 3D position tracking
 * - Distance measurement
 * - Bed proximity detection
 */
class DepthProcessor {
public:
    DepthProcessor();
    ~DepthProcessor();

    /**
     * Initialize with depth frame dimensions
     */
    void init(int width, int height);

    /**
     * Update with new depth frame
     * @param depth_data DEPTH16 data (uint16_t, values in millimeters)
     * @param width Frame width
     * @param height Frame height
     */
    void updateDepthMap(const uint16_t* depth_data, int width, int height);

    /**
     * Get depth value at pixel coordinates
     * @return Depth in meters, or -1 if invalid
     */
    float getDepthAt(int x, int y) const;

    /**
     * Get depth at normalized coordinates (0-1)
     */
    float getDepthAtNormalized(float norm_x, float norm_y) const;

    /**
     * Calculate depth statistics within a bounding box
     */
    DepthStats calculateStats(const BoundingBox& bbox) const;

    /**
     * Estimate 3D position of person based on bounding box
     * @param person_bbox Person bounding box from YOLO (normalized coords)
     * @param rgb_width RGB frame width (for coordinate scaling)
     * @param rgb_height RGB frame height
     */
    Position3D estimate3DPosition(
        const BoundingBox& person_bbox,
        int rgb_width,
        int rgb_height
    ) const;

    /**
     * Detect fall using vertical drop analysis
     * @param person_bbox Current person bounding box
     * @param rgb_width RGB frame width
     * @param rgb_height RGB frame height
     */
    DepthFallResult detectFall(
        const BoundingBox& person_bbox,
        int rgb_width,
        int rgb_height
    );

    /**
     * Analyze depth-based motion
     */
    DepthMotionResult analyzeMotion(
        const BoundingBox& person_bbox,
        int rgb_width,
        int rgb_height
    );

    /**
     * Configure bed region for proximity detection
     * @param center 3D position of bed center
     * @param radius_meters Radius of bed zone
     */
    void setBedRegion(const Position3D& center, float radius_meters);

    /**
     * Get average distance to person (last measurement)
     */
    float getAverageDistance() const { return last_distance_; }

    /**
     * Check if depth data is available
     */
    bool hasDepthData() const { return initialized_ && !depth_map_.empty(); }

    /**
     * Reset state (call when patient changes)
     */
    void reset();

private:
    bool initialized_ = false;
    int width_ = 0;
    int height_ = 0;

    // Current depth map
    std::vector<uint16_t> depth_map_;

    // Temporal tracking for fall detection
    struct PositionSample {
        Position3D position;
        int64_t timestamp_ms;
    };
    std::deque<PositionSample> position_history_;
    static const size_t MAX_HISTORY_SIZE = 30;  // ~1 second at 30fps

    // Fall detection thresholds
    float fall_drop_threshold_ = 0.5f;     // 0.5m drop = fall
    float fall_velocity_threshold_ = 1.5f;  // 1.5 m/s = fall speed
    int64_t fall_time_window_ms_ = 1000;    // 1 second window

    // Bed zone configuration
    Position3D bed_center_ = {0, 0, 2.0f};  // Default: 2m from camera
    float bed_radius_ = 1.5f;                // 1.5m radius

    // Camera intrinsics (approximate for typical ToF sensor)
    float focal_length_x_ = 500.0f;
    float focal_length_y_ = 500.0f;
    float principal_x_ = 0.0f;  // Set on init
    float principal_y_ = 0.0f;

    // Last measurements
    float last_distance_ = 0.0f;
    Position3D last_position_ = {0, 0, 0};

    // Helper functions
    float medianDepthInRegion(int x1, int y1, int x2, int y2) const;
    void updatePositionHistory(const Position3D& pos);
    float calculateVerticalDrop() const;
    float calculateDropVelocity() const;
    int64_t getCurrentTimeMs() const;
};

} // namespace triage
