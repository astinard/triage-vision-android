#pragma once

#include <string>
#include <vector>

#ifdef HAVE_LLAMA
#include <llama.h>
#endif

#ifdef HAVE_MTMD
#include <mtmd.h>
#include <mtmd-helper.h>
#endif

namespace triage {

struct VLMObservation {
    std::string position;
    std::string alertness;
    std::string movement_level;
    std::vector<std::string> equipment_visible;
    std::vector<std::string> concerns;
    std::string comfort_assessment;
    std::string chart_note;
    std::string raw_output;
    bool success;
    std::string error;
};

class VLMInference {
public:
    VLMInference();
    ~VLMInference();

    /**
     * Initialize VLM with model files
     * @param model_path Path to GGUF model file
     * @param mmproj_path Path to multimodal projector file (for vision)
     * @param n_threads Number of CPU threads
     * @param n_gpu_layers Layers to offload to GPU (0 = CPU only)
     * @return true on success
     */
    bool init(const std::string& model_path,
              const std::string& mmproj_path = "",
              int n_threads = 4,
              int n_gpu_layers = 0);

    /**
     * Run inference on an image
     * @param pixels RGBA pixel data
     * @param width Image width
     * @param height Image height
     * @param prompt Analysis prompt
     * @return Observation result
     */
    VLMObservation analyze(const uint8_t* pixels, int width, int height,
                           const std::string& prompt);

    /**
     * Check if model is loaded
     */
    bool isInitialized() const { return initialized_; }

    /**
     * Get model info
     */
    std::string getModelInfo() const;

    /**
     * Cleanup resources
     */
    void cleanup();

    /**
     * Get default analysis prompt
     */
    static std::string getDefaultPrompt();

private:
    bool initialized_ = false;
    bool vision_enabled_ = false;

#ifdef HAVE_LLAMA
    llama_model* model_ = nullptr;
    llama_context* llama_ctx_ = nullptr;
#endif

#ifdef HAVE_MTMD
    mtmd_context* mtmd_ctx_ = nullptr;
#endif

    int n_threads_ = 4;
    int n_ctx_ = 2048;
    int max_tokens_ = 512;
    int n_batch_ = 512;

    std::string generateResponseWithImage(const uint8_t* pixels, int width, int height,
                                           const std::string& prompt);
    std::string generateResponseTextOnly(const std::string& prompt);
    VLMObservation parseResponse(const std::string& response);
};

} // namespace triage
