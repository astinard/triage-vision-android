#include "image_processor.h"
#include <cmath>
#include <algorithm>

namespace triage {

std::vector<uint8_t> ImageProcessor::resize(const uint8_t* src, int src_width, int src_height,
                                             int dst_width, int dst_height, int channels) {
    std::vector<uint8_t> dst(dst_width * dst_height * channels);

    float x_ratio = static_cast<float>(src_width) / dst_width;
    float y_ratio = static_cast<float>(src_height) / dst_height;

    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, src_width - 1);
            int y1 = std::min(y0 + 1, src_height - 1);

            float x_diff = src_x - x0;
            float y_diff = src_y - y0;

            for (int c = 0; c < channels; c++) {
                float p00 = src[(y0 * src_width + x0) * channels + c];
                float p10 = src[(y0 * src_width + x1) * channels + c];
                float p01 = src[(y1 * src_width + x0) * channels + c];
                float p11 = src[(y1 * src_width + x1) * channels + c];

                float value = p00 * (1 - x_diff) * (1 - y_diff) +
                              p10 * x_diff * (1 - y_diff) +
                              p01 * (1 - x_diff) * y_diff +
                              p11 * x_diff * y_diff;

                dst[(y * dst_width + x) * channels + c] = static_cast<uint8_t>(
                    std::clamp(value, 0.0f, 255.0f));
            }
        }
    }

    return dst;
}

std::vector<uint8_t> ImageProcessor::rgbaToRgb(const uint8_t* rgba, int width, int height) {
    std::vector<uint8_t> rgb(width * height * 3);

    for (int i = 0; i < width * height; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }

    return rgb;
}

std::vector<float> ImageProcessor::normalizeToFloat(const uint8_t* src, int width, int height,
                                                     int channels) {
    std::vector<float> dst(width * height * channels);

    for (int i = 0; i < width * height * channels; i++) {
        dst[i] = src[i] / 255.0f;
    }

    return dst;
}

std::vector<float> ImageProcessor::normalizeImageNet(const uint8_t* src, int width, int height) {
    // ImageNet mean and std
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    std::vector<float> dst(width * height * 3);

    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            float value = src[i * 3 + c] / 255.0f;
            dst[i * 3 + c] = (value - mean[c]) / std[c];
        }
    }

    return dst;
}

std::vector<uint8_t> ImageProcessor::centerCrop(const uint8_t* src, int src_width, int src_height,
                                                 int crop_size, int channels) {
    std::vector<uint8_t> dst(crop_size * crop_size * channels);

    int offset_x = (src_width - crop_size) / 2;
    int offset_y = (src_height - crop_size) / 2;

    // Clamp offsets
    offset_x = std::max(0, offset_x);
    offset_y = std::max(0, offset_y);

    int actual_crop_w = std::min(crop_size, src_width - offset_x);
    int actual_crop_h = std::min(crop_size, src_height - offset_y);

    for (int y = 0; y < actual_crop_h; y++) {
        for (int x = 0; x < actual_crop_w; x++) {
            int src_idx = ((y + offset_y) * src_width + (x + offset_x)) * channels;
            int dst_idx = (y * crop_size + x) * channels;

            for (int c = 0; c < channels; c++) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }

    return dst;
}

std::vector<uint8_t> ImageProcessor::grayToRgb(const uint8_t* gray, int width, int height) {
    std::vector<uint8_t> rgb(width * height * 3);

    for (int i = 0; i < width * height; i++) {
        rgb[i * 3 + 0] = gray[i];
        rgb[i * 3 + 1] = gray[i];
        rgb[i * 3 + 2] = gray[i];
    }

    return rgb;
}

} // namespace triage
