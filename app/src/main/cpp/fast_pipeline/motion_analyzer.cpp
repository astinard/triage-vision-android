#include "motion_analyzer.h"
#include <android/log.h>
#include <cmath>
#include <numeric>
#include <algorithm>

#define LOG_TAG "MotionAnalyzer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace triage {

MotionAnalyzer::MotionAnalyzer() {
    reset();
}

MotionAnalyzer::~MotionAnalyzer() {
}

void MotionAnalyzer::init(float stillness_threshold, int history_frames) {
    stillness_threshold_ = stillness_threshold;
    history_frames_ = history_frames;
    initialized_ = true;
    reset();
    LOGI("Motion analyzer initialized (threshold=%.2f, history=%d)",
         stillness_threshold, history_frames);
}

MotionState MotionAnalyzer::analyze(const uint8_t* pixels, int width, int height) {
    MotionState state;
    state.motion_level = 0.0f;
    state.is_still = true;

    auto now = std::chrono::system_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    // First frame - just store it
    if (prev_frame_.empty() || prev_width_ != width || prev_height_ != height) {
        int size = width * height * 4; // RGBA
        prev_frame_.resize(size);
        std::copy(pixels, pixels + size, prev_frame_.begin());
        prev_width_ = width;
        prev_height_ = height;

        state.last_motion_timestamp = now_ms;
        state.stillness_duration = 0;
        last_motion_time_ = now_ms;
        stillness_start_time_ = now_ms;

        return state;
    }

    // Calculate motion between frames
    float frame_diff = calculateFrameDifference(pixels, prev_frame_.data(), width, height);

    // Update motion history
    motion_history_.push_back(frame_diff);
    if (motion_history_.size() > history_frames_) {
        motion_history_.pop_front();
    }

    // Calculate average motion level
    if (!motion_history_.empty()) {
        current_motion_level_ = std::accumulate(motion_history_.begin(),
                                                  motion_history_.end(), 0.0f)
                                 / motion_history_.size();
    }

    // Update timing
    bool is_motion = current_motion_level_ > stillness_threshold_;
    if (is_motion) {
        last_motion_time_ = now_ms;
        stillness_start_time_ = now_ms;
    }

    // Store current frame for next comparison
    std::copy(pixels, pixels + width * height * 4, prev_frame_.begin());

    // Build state
    state.motion_level = current_motion_level_;
    state.last_motion_timestamp = last_motion_time_;
    state.stillness_duration = now_ms - stillness_start_time_;
    state.is_still = !is_motion;

    return state;
}

float MotionAnalyzer::calculateFrameDifference(const uint8_t* current,
                                                const uint8_t* previous,
                                                int width, int height) {
    // Downsample for efficiency
    int step = 4; // Check every 4th pixel
    int sample_count = 0;
    float total_diff = 0.0f;

    for (int y = 0; y < height; y += step) {
        for (int x = 0; x < width; x += step) {
            int idx = (y * width + x) * 4;

            // Calculate luminance difference
            float curr_lum = 0.299f * current[idx] + 0.587f * current[idx+1] + 0.114f * current[idx+2];
            float prev_lum = 0.299f * previous[idx] + 0.587f * previous[idx+1] + 0.114f * previous[idx+2];

            float diff = std::abs(curr_lum - prev_lum) / 255.0f;
            total_diff += diff;
            sample_count++;
        }
    }

    if (sample_count == 0) return 0.0f;

    // Normalize to 0-1 range
    float avg_diff = total_diff / sample_count;

    // Apply sensitivity curve (small changes amplified)
    return std::min(1.0f, avg_diff * 5.0f);
}

float MotionAnalyzer::calculateOpticalFlowMagnitude(const uint8_t* current,
                                                     const uint8_t* previous,
                                                     int width, int height) {
    // Simplified optical flow using block matching
    // For production, consider using a proper optical flow algorithm

    int block_size = 16;
    int search_range = 8;
    float total_magnitude = 0.0f;
    int block_count = 0;

    for (int by = search_range; by < height - block_size - search_range; by += block_size) {
        for (int bx = search_range; bx < width - block_size - search_range; bx += block_size) {
            // Find best match in search window
            float best_match = 1e9f;
            int best_dx = 0, best_dy = 0;

            for (int dy = -search_range; dy <= search_range; dy += 2) {
                for (int dx = -search_range; dx <= search_range; dx += 2) {
                    float sad = 0; // Sum of absolute differences

                    for (int py = 0; py < block_size; py += 2) {
                        for (int px = 0; px < block_size; px += 2) {
                            int curr_idx = ((by + py) * width + (bx + px)) * 4;
                            int prev_idx = ((by + py + dy) * width + (bx + px + dx)) * 4;

                            sad += std::abs((int)current[curr_idx] - (int)previous[prev_idx]);
                        }
                    }

                    if (sad < best_match) {
                        best_match = sad;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            float magnitude = std::sqrt(best_dx * best_dx + best_dy * best_dy);
            total_magnitude += magnitude;
            block_count++;
        }
    }

    if (block_count == 0) return 0.0f;

    // Normalize
    return std::min(1.0f, (total_magnitude / block_count) / search_range);
}

int64_t MotionAnalyzer::getSecondsSinceMotion() const {
    auto now = std::chrono::system_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    return (now_ms - last_motion_time_) / 1000;
}

bool MotionAnalyzer::shouldAlertStillness(int threshold_seconds) const {
    return getSecondsSinceMotion() >= threshold_seconds;
}

void MotionAnalyzer::reset() {
    prev_frame_.clear();
    prev_width_ = 0;
    prev_height_ = 0;
    motion_history_.clear();
    current_motion_level_ = 0.0f;

    auto now = std::chrono::system_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    last_motion_time_ = now_ms;
    stillness_start_time_ = now_ms;
}

} // namespace triage
