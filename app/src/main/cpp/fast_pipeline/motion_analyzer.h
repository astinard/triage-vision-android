#pragma once

#include <vector>
#include <chrono>
#include <deque>

namespace triage {

struct MotionState {
    float motion_level;           // 0.0 (still) to 1.0 (active)
    int64_t last_motion_timestamp; // ms since epoch
    int64_t stillness_duration;   // ms of continuous stillness
    bool is_still;
};

class MotionAnalyzer {
public:
    MotionAnalyzer();
    ~MotionAnalyzer();

    /**
     * Initialize with configuration
     * @param stillness_threshold Motion level below which is considered "still"
     * @param history_frames Number of frames to keep for motion history
     */
    void init(float stillness_threshold = 0.05f, int history_frames = 30);

    /**
     * Analyze motion between current and previous frame
     * @param pixels Current frame RGBA data
     * @param width Frame width
     * @param height Frame height
     * @return Current motion state
     */
    MotionState analyze(const uint8_t* pixels, int width, int height);

    /**
     * Get current motion level (0.0-1.0)
     */
    float getMotionLevel() const { return current_motion_level_; }

    /**
     * Get seconds since last significant motion
     */
    int64_t getSecondsSinceMotion() const;

    /**
     * Check if prolonged stillness alert should trigger
     * @param threshold_seconds Seconds of stillness to trigger alert
     */
    bool shouldAlertStillness(int threshold_seconds) const;

    /**
     * Reset motion history (e.g., when starting new session)
     */
    void reset();

private:
    bool initialized_ = false;
    float stillness_threshold_ = 0.05f;
    int history_frames_ = 30;

    // Previous frame for comparison
    std::vector<uint8_t> prev_frame_;
    int prev_width_ = 0;
    int prev_height_ = 0;

    // Motion history
    std::deque<float> motion_history_;
    float current_motion_level_ = 0.0f;

    // Timing
    int64_t last_motion_time_ = 0;
    int64_t stillness_start_time_ = 0;

    float calculateFrameDifference(const uint8_t* current, const uint8_t* previous,
                                    int width, int height);
    float calculateOpticalFlowMagnitude(const uint8_t* current, const uint8_t* previous,
                                         int width, int height);
};

} // namespace triage
