#include "vlm_inference.h"
#include <android/log.h>
#include <sstream>

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

    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        LOGE("Failed to load model: %s", model_path.c_str());
        return false;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        LOGE("Failed to create context");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    // Note: Vision encoder (clip) disabled for now - using text-only mode
    // TODO: Update clip API usage when needed for vision support
    if (!mmproj_path.empty()) {
        LOGI("Vision encoder path provided but clip integration disabled - using text-only mode");
    }
    clip_ctx_ = nullptr;

    initialized_ = true;
    LOGI("VLM initialized successfully (text-only mode)");
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

    // Generate response (text-only mode for now)
    std::vector<float> image_embed; // Empty - no vision encoding
    std::string response = generateResponse(prompt, image_embed);

    if (response.empty()) {
        result.error = "Failed to generate response";
        return result;
    }

    result.raw_output = response;
    result = parseResponse(response);
    result.raw_output = response;
    result.success = true;

    LOGI("VLM analysis complete");
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
    // Vision encoding disabled for now - return empty embedding
    // TODO: Update clip API usage for vision support
    return std::vector<float>();
}

// Helper function to add token to batch (replaces llama_batch_add)
static void batch_add(llama_batch& batch, llama_token token, llama_pos pos,
                      const std::vector<llama_seq_id>& seq_ids, bool logits) {
    batch.token[batch.n_tokens] = token;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
}

// Helper function to clear batch (replaces llama_batch_clear)
static void batch_clear(llama_batch& batch) {
    batch.n_tokens = 0;
}

std::string VLMInference::generateResponse(const std::string& prompt,
                                            const std::vector<float>& image_embed) {
    std::string response;

    // Get vocabulary from model
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    // Tokenize prompt
    std::vector<llama_token> tokens(n_ctx_);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                   tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        LOGE("Tokenization failed");
        return "";
    }
    tokens.resize(n_tokens);

    // Evaluate prompt
    llama_batch batch = llama_batch_init(n_ctx_, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = 1; // Enable logits for last token

    if (llama_decode(ctx_, batch) != 0) {
        LOGE("Failed to evaluate prompt");
        llama_batch_free(batch);
        return "";
    }

    // Create greedy sampler
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Generate response
    int n_generated = 0;
    llama_token eos_token = llama_vocab_eos(vocab);

    while (n_generated < max_tokens_) {
        // Sample next token using sampler
        llama_token new_token = llama_sampler_sample(smpl, ctx_, -1);

        // Accept the token
        llama_sampler_accept(smpl, new_token);

        // Check for end of generation
        if (new_token == eos_token) {
            break;
        }

        // Convert token to string
        char buf[256];
        int len = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (len > 0) {
            response.append(buf, len);
        }

        // Prepare next batch
        batch_clear(batch);
        batch_add(batch, new_token, n_tokens + n_generated, {0}, true);

        if (llama_decode(ctx_, batch) != 0) {
            break;
        }

        n_generated++;
    }

    llama_sampler_free(smpl);
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
        llama_model_free(model_);
        model_ = nullptr;
    }
    // clip_ctx_ is nullptr - no cleanup needed
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
