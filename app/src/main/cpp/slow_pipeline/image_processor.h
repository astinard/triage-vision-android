#pragma once

#include <cstdint>
#include <vector>

namespace triage {

/**
 * Image preprocessing utilities for VLM inference
 */
class ImageProcessor {
public:
    /**
     * Resize image using bilinear interpolation
     */
    static std::vector<uint8_t> resize(const uint8_t* src, int src_width, int src_height,
                                        int dst_width, int dst_height, int channels = 4);

    /**
     * Convert RGBA to RGB
     */
    static std::vector<uint8_t> rgbaToRgb(const uint8_t* rgba, int width, int height);

    /**
     * Normalize image to float [0, 1]
     */
    static std::vector<float> normalizeToFloat(const uint8_t* src, int width, int height,
                                                int channels = 3);

    /**
     * Apply ImageNet normalization (mean subtract, std divide)
     */
    static std::vector<float> normalizeImageNet(const uint8_t* src, int width, int height);

    /**
     * Center crop image
     */
    static std::vector<uint8_t> centerCrop(const uint8_t* src, int src_width, int src_height,
                                            int crop_size, int channels = 4);

    /**
     * Convert grayscale to RGB
     */
    static std::vector<uint8_t> grayToRgb(const uint8_t* gray, int width, int height);
};

} // namespace triage
