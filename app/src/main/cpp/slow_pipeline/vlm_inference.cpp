#include "vlm_inference.h"
#include <android/log.h>
#include <sstream>
#include <cstring>

#define LOG_TAG "VLMInference"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

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
#if defined(HAVE_LLAMA) && defined(HAVE_MTMD)
    LOGI("Initializing VLM with mtmd from: %s", model_path.c_str());
    LOGI("mmproj path: %s", mmproj_path.c_str());
    n_threads_ = n_threads;

    // Initialize llama backend
    llama_backend_init();

    // Load text model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        LOGE("Failed to load text model: %s", model_path.c_str());
        return false;
    }
    LOGI("Text model loaded successfully");

    // Create llama context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.n_batch = n_batch_;

    llama_ctx_ = llama_init_from_model(model_, ctx_params);
    if (!llama_ctx_) {
        LOGE("Failed to create llama context");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }
    LOGI("Llama context created");

    // Initialize mtmd context for vision support
    if (!mmproj_path.empty()) {
        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = (n_gpu_layers > 0);
        mtmd_params.n_threads = n_threads;
        mtmd_params.print_timings = false;
        mtmd_params.warmup = false; // Skip warmup for faster init

        mtmd_ctx_ = mtmd_init_from_file(mmproj_path.c_str(), model_, mtmd_params);
        if (mtmd_ctx_) {
            vision_enabled_ = mtmd_support_vision(mtmd_ctx_);
            if (vision_enabled_) {
                LOGI("Vision support enabled via mtmd");
            } else {
                LOGW("mtmd initialized but vision not supported by this model");
            }
        } else {
            LOGW("Failed to initialize mtmd from: %s", mmproj_path.c_str());
            LOGW("Continuing in text-only mode");
        }
    } else {
        LOGI("No mmproj path provided, running in text-only mode");
    }

    initialized_ = true;
    LOGI("VLM initialized successfully (vision=%s)", vision_enabled_ ? "enabled" : "disabled");
    return true;

#elif defined(HAVE_LLAMA)
    // Fallback: llama available but no mtmd
    LOGI("Initializing VLM (text-only, no mtmd) from: %s", model_path.c_str());
    n_threads_ = n_threads;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        LOGE("Failed to load model: %s", model_path.c_str());
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    llama_ctx_ = llama_init_from_model(model_, ctx_params);
    if (!llama_ctx_) {
        LOGE("Failed to create context");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    initialized_ = true;
    LOGI("VLM initialized (text-only mode, mtmd not available)");
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

#if defined(HAVE_LLAMA)
    if (!initialized_) {
        result.error = "VLM not initialized";
        return result;
    }

    LOGI("Running VLM analysis (%dx%d)", width, height);

    std::string response;

#ifdef HAVE_MTMD
    if (vision_enabled_ && pixels != nullptr && width > 0 && height > 0) {
        response = generateResponseWithImage(pixels, width, height, prompt);
    } else {
        response = generateResponseTextOnly(prompt);
    }
#else
    response = generateResponseTextOnly(prompt);
#endif

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

#ifdef HAVE_MTMD
std::string VLMInference::generateResponseWithImage(const uint8_t* pixels, int width, int height,
                                                     const std::string& prompt) {
    std::string response;

    LOGI("Generating response with image (%dx%d)", width, height);

    // Convert RGBA to RGB (mtmd expects RGB)
    std::vector<unsigned char> rgb_data(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        rgb_data[i * 3 + 0] = pixels[i * 4 + 0]; // R
        rgb_data[i * 3 + 1] = pixels[i * 4 + 1]; // G
        rgb_data[i * 3 + 2] = pixels[i * 4 + 2]; // B
        // Alpha (pixels[i * 4 + 3]) is discarded
    }

    // Create bitmap from RGB data
    mtmd_bitmap* bitmap = mtmd_bitmap_init(width, height, rgb_data.data());
    if (!bitmap) {
        LOGE("Failed to create mtmd bitmap");
        return "";
    }

    // Build prompt with media marker
    // The marker will be replaced with the image tokens
    std::string full_prompt = std::string(mtmd_default_marker()) + "\n" + prompt;

    // Prepare input text
    mtmd_input_text input_text;
    input_text.text = full_prompt.c_str();
    input_text.add_special = true;
    input_text.parse_special = true;

    // Tokenize with image
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[] = { bitmap };

    int32_t ret = mtmd_tokenize(mtmd_ctx_, chunks, &input_text, bitmaps, 1);
    if (ret != 0) {
        LOGE("mtmd_tokenize failed with code: %d", ret);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return "";
    }

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    LOGI("Tokenized into %zu chunks", n_chunks);

    // Clear KV cache
    llama_memory_clear(llama_get_memory(llama_ctx_), true);

    // Evaluate chunks (text and image)
    llama_pos n_past = 0;
    ret = mtmd_helper_eval_chunks(mtmd_ctx_, llama_ctx_, chunks, n_past,
                                   0, n_batch_, true, &n_past);
    if (ret != 0) {
        LOGE("mtmd_helper_eval_chunks failed with code: %d", ret);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bitmap);
        return "";
    }

    LOGI("Chunks evaluated, n_past=%d", (int)n_past);

    // Get vocabulary
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    llama_token eos_token = llama_vocab_eos(vocab);

    // Create sampler
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Generate tokens
    int n_generated = 0;
    llama_batch batch = llama_batch_init(1, 0, 1);

    while (n_generated < max_tokens_) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(smpl, llama_ctx_, -1);
        llama_sampler_accept(smpl, new_token);

        // Check for end
        if (new_token == eos_token) {
            LOGI("EOS token reached after %d tokens", n_generated);
            break;
        }

        // Convert token to text
        char buf[256];
        int len = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (len > 0) {
            response.append(buf, len);
        }

        // Prepare next decode
        batch.n_tokens = 0;
        batch.token[0] = new_token;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        if (llama_decode(llama_ctx_, batch) != 0) {
            LOGE("llama_decode failed during generation");
            break;
        }

        n_past++;
        n_generated++;
    }

    LOGI("Generated %d tokens", n_generated);

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);

    return response;
}
#endif

#ifdef HAVE_LLAMA
std::string VLMInference::generateResponseTextOnly(const std::string& prompt) {
    std::string response;

    LOGI("Generating response (text-only mode)");

    // Get vocabulary
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

    // Clear KV cache
    llama_memory_clear(llama_get_memory(llama_ctx_), true);

    // Evaluate prompt
    llama_batch batch = llama_batch_init(n_ctx_, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = 0;
        batch.n_tokens++;
    }
    batch.logits[batch.n_tokens - 1] = 1; // Enable logits for last token

    if (llama_decode(llama_ctx_, batch) != 0) {
        LOGE("Failed to evaluate prompt");
        llama_batch_free(batch);
        return "";
    }

    // Create sampler
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Generate response
    int n_generated = 0;
    llama_token eos_token = llama_vocab_eos(vocab);

    while (n_generated < max_tokens_) {
        llama_token new_token = llama_sampler_sample(smpl, llama_ctx_, -1);
        llama_sampler_accept(smpl, new_token);

        if (new_token == eos_token) {
            break;
        }

        char buf[256];
        int len = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (len > 0) {
            response.append(buf, len);
        }

        // Prepare next batch
        batch.n_tokens = 0;
        batch.token[0] = new_token;
        batch.pos[0] = n_tokens + n_generated;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        if (llama_decode(llama_ctx_, batch) != 0) {
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
    info << "Model loaded";
    info << ", Context: " << n_ctx_;
    info << ", Threads: " << n_threads_;
#ifdef HAVE_MTMD
    info << ", Vision: " << (vision_enabled_ ? "enabled" : "disabled");
#else
    info << ", Vision: not available (no mtmd)";
#endif
    return info.str();
#else
    return "llama.cpp not available";
#endif
}

void VLMInference::cleanup() {
#ifdef HAVE_MTMD
    if (mtmd_ctx_) {
        mtmd_free(mtmd_ctx_);
        mtmd_ctx_ = nullptr;
    }
#endif

#ifdef HAVE_LLAMA
    if (llama_ctx_) {
        llama_free(llama_ctx_);
        llama_ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    llama_backend_free();
    LOGI("VLM cleaned up");
#endif

    initialized_ = false;
    vision_enabled_ = false;
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
