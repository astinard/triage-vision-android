#include "vlm_inference.h"
#include <android/log.h>
#include <sstream>
#include <regex>

#define LOG_TAG "VLMInference"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace triage {

VLMInference::VLMInference() {
}

VLMInference::~VLMInference() {
    cleanup();
}

bool VLMInference::init(const std::string& model_path,
                        const std::string& mmproj_path,
                        int n_threads,
                        int n_gpu_layers) {
#ifdef HAVE_LLAMA
    LOGI("Initializing VLM from: %s", model_path.c_str());
    n_threads_ = n_threads;

    // Initialize llama backend
    llama_backend_init();

    // Load model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    model_ = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model_) {
        LOGE("Failed to load model: %s", model_path.c_str());
        return false;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    ctx_ = llama_new_context_with_model(model_, ctx_params);
    if (!ctx_) {
        LOGE("Failed to create context");
        llama_free_model(model_);
        model_ = nullptr;
        return false;
    }

    // Load vision encoder if provided
    if (!mmproj_path.empty()) {
        clip_ctx_ = clip_model_load(mmproj_path.c_str(), 1);
        if (!clip_ctx_) {
            LOGE("Failed to load vision encoder: %s", mmproj_path.c_str());
            // Continue without vision - will use text-only
        }
    }

    initialized_ = true;
    LOGI("VLM initialized successfully");
    return true;
#else
    LOGE("llama.cpp not available - VLM disabled");
    return false;
#endif
}

VLMObservation VLMInference::analyze(const uint8_t* pixels, int width, int height,
                                      const std::string& prompt) {
    VLMObservation result;
    result.success = false;

#ifdef HAVE_LLAMA
    if (!initialized_) {
        result.error = "VLM not initialized";
        return result;
    }

    LOGI("Running VLM analysis (%dx%d)", width, height);

    try {
        // Encode image if vision encoder available
        std::vector<float> image_embed;
        if (clip_ctx_) {
            image_embed = encodeImage(pixels, width, height);
        }

        // Generate response
        std::string response = generateResponse(prompt, image_embed);

        result.raw_output = response;
        result = parseResponse(response);
        result.raw_output = response;
        result.success = true;

        LOGI("VLM analysis complete");

    } catch (const std::exception& e) {
        LOGE("VLM inference error: %s", e.what());
        result.error = e.what();
    }
#else
    result.error = "llama.cpp not available";

    // Return placeholder for testing
    result.position = "unknown";
    result.alertness = "unknown";
    result.movement_level = "unknown";
    result.comfort_assessment = "unknown";
    result.chart_note = "VLM inference not available - placeholder observation";
    result.success = true; // Allow app to continue
#endif

    return result;
}

#ifdef HAVE_LLAMA
std::vector<float> VLMInference::encodeImage(const uint8_t* pixels, int width, int height) {
    std::vector<float> embedding;

    if (!clip_ctx_) return embedding;

    // Create clip image from pixels
    clip_image_u8 img;
    img.nx = width;
    img.ny = height;
    img.data.resize(width * height * 3);

    // Convert RGBA to RGB
    for (int i = 0; i < width * height; i++) {
        img.data[i*3 + 0] = pixels[i*4 + 0];
        img.data[i*3 + 1] = pixels[i*4 + 1];
        img.data[i*3 + 2] = pixels[i*4 + 2];
    }

    // Preprocess and encode
    clip_image_f32 img_res;
    clip_image_preprocess(clip_ctx_, &img, &img_res, true);

    int n_embed = clip_n_patches(clip_ctx_) * clip_n_mmproj_embd(clip_ctx_);
    embedding.resize(n_embed);

    clip_image_encode(clip_ctx_, n_threads_, &img_res, embedding.data());

    return embedding;
}

std::string VLMInference::generateResponse(const std::string& prompt,
                                            const std::vector<float>& image_embed) {
    std::string response;

    // Tokenize prompt
    std::vector<llama_token> tokens(n_ctx_);
    int n_tokens = llama_tokenize(model_, prompt.c_str(), prompt.length(),
                                   tokens.data(), tokens.size(), true, false);
    tokens.resize(n_tokens);

    // TODO: Inject image embeddings if available
    // This requires proper handling based on the specific VLM architecture

    // Evaluate prompt
    llama_batch batch = llama_batch_init(n_ctx_, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx_, batch) != 0) {
        LOGE("Failed to evaluate prompt");
        llama_batch_free(batch);
        return "";
    }

    // Generate response
    int n_generated = 0;
    while (n_generated < max_tokens_) {
        // Sample next token
        auto logits = llama_get_logits_ith(ctx_, batch.n_tokens - 1);
        auto n_vocab = llama_n_vocab(model_);

        llama_token new_token = llama_sample_token_greedy(ctx_, logits);

        // Check for end of generation
        if (new_token == llama_token_eos(model_)) {
            break;
        }

        // Convert token to string
        char buf[256];
        int len = llama_token_to_piece(model_, new_token, buf, sizeof(buf), false);
        if (len > 0) {
            response.append(buf, len);
        }

        // Prepare next batch
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, n_tokens + n_generated, {0}, true);

        if (llama_decode(ctx_, batch) != 0) {
            break;
        }

        n_generated++;
    }

    llama_batch_free(batch);
    return response;
}
#endif

VLMObservation VLMInference::parseResponse(const std::string& response) {
    VLMObservation obs;

    // Try to find JSON in response
    size_t json_start = response.find('{');
    size_t json_end = response.rfind('}');

    if (json_start != std::string::npos && json_end != std::string::npos && json_end > json_start) {
        std::string json_str = response.substr(json_start, json_end - json_start + 1);

        // Simple JSON parsing (production should use proper JSON library)
        auto extract = [&json_str](const std::string& key) -> std::string {
            std::string search = "\"" + key + "\"";
            size_t pos = json_str.find(search);
            if (pos == std::string::npos) return "";

            pos = json_str.find(':', pos);
            if (pos == std::string::npos) return "";

            pos = json_str.find('"', pos + 1);
            if (pos == std::string::npos) return "";

            size_t end = json_str.find('"', pos + 1);
            if (end == std::string::npos) return "";

            return json_str.substr(pos + 1, end - pos - 1);
        };

        obs.position = extract("position");
        obs.alertness = extract("alertness");
        obs.movement_level = extract("movement_level");
        obs.comfort_assessment = extract("comfort_assessment");
        obs.chart_note = extract("chart_note");

        // Extract arrays (simplified)
        // TODO: Proper array parsing
    }

    // Fallback to raw response as chart note
    if (obs.chart_note.empty()) {
        obs.chart_note = response;
    }

    // Set defaults for empty fields
    if (obs.position.empty()) obs.position = "unknown";
    if (obs.alertness.empty()) obs.alertness = "unknown";
    if (obs.movement_level.empty()) obs.movement_level = "unknown";
    if (obs.comfort_assessment.empty()) obs.comfort_assessment = "unknown";

    return obs;
}

std::string VLMInference::getModelInfo() const {
#ifdef HAVE_LLAMA
    if (!model_) return "Model not loaded";

    std::ostringstream info;
    info << "Model: " << llama_model_desc(model_, nullptr, 0);
    info << ", Context: " << n_ctx_;
    info << ", Threads: " << n_threads_;
    return info.str();
#else
    return "llama.cpp not available";
#endif
}

void VLMInference::cleanup() {
#ifdef HAVE_LLAMA
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_free_model(model_);
        model_ = nullptr;
    }
    if (clip_ctx_) {
        clip_free(clip_ctx_);
        clip_ctx_ = nullptr;
    }
    llama_backend_free();
    initialized_ = false;
    LOGI("VLM cleaned up");
#endif
}

std::string VLMInference::getDefaultPrompt() {
    return R"(Analyze this patient monitoring image. Describe:
1. Patient position (lying_supine, lying_left_lateral, lying_right_lateral, sitting, standing)
2. Alertness level (awake, sleeping, drowsy, eyes_closed, unresponsive)
3. Movement level (none, minimal, moderate, active)
4. Any visible medical equipment (iv_line, pulse_oximeter, nasal_cannula, feeding_tube, catheter, monitor_leads)
5. Any concerns or notable observations
6. General patient comfort assessment (comfortable, restless, in_distress, pain_indicated)

Respond ONLY with valid JSON:
{"position": "", "alertness": "", "movement_level": "", "equipment_visible": [], "concerns": [], "comfort_assessment": "", "chart_note": ""})";
}

} // namespace triage
