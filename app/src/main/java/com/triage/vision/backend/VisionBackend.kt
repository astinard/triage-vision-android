package com.triage.vision.backend

import android.content.Context
import android.graphics.Bitmap

/**
 * Abstract interface for vision processing backends.
 *
 * This abstraction allows the app to run on different hardware accelerators:
 * - ONNX Runtime with NNAPI (current default for most devices)
 * - NCNN (optimized for ARM, used for YOLO)
 * - QNN (Qualcomm Neural Network SDK for QCM6490 NPU)
 * - CPU fallback
 *
 * The backend registry automatically selects the best available backend
 * based on device capabilities.
 */
interface VisionBackend {

    /**
     * Backend type identifier
     */
    val type: BackendType

    /**
     * Human-readable name for logging/UI
     */
    val name: String

    /**
     * Initialize the backend with the given model
     *
     * @param context Android context for asset access
     * @param config Backend-specific configuration
     * @return true if initialization successful
     */
    suspend fun initialize(context: Context, config: BackendConfig): Boolean

    /**
     * Check if backend is ready for inference
     */
    fun isReady(): Boolean

    /**
     * Get backend capabilities
     */
    fun getCapabilities(): BackendCapabilities

    /**
     * Process an image and return embeddings
     *
     * @param bitmap Input image
     * @return Float array of embeddings, or null on error
     */
    suspend fun processImage(bitmap: Bitmap): FloatArray?

    /**
     * Process an image and return classification scores for given labels
     *
     * @param bitmap Input image
     * @param labelEmbeddings Pre-computed label embeddings [numLabels x embeddingDim]
     * @return Float array of scores per label, or null on error
     */
    suspend fun classify(bitmap: Bitmap, labelEmbeddings: Array<FloatArray>): FloatArray?

    /**
     * Get inference timing statistics
     */
    fun getStats(): InferenceStats

    /**
     * Release all resources
     */
    fun close()
}

/**
 * Backend type enumeration
 */
enum class BackendType(val priority: Int) {
    QNN(1),      // Qualcomm Neural Network - highest priority on supported devices
    NNAPI(2),    // Android Neural Networks API
    ONNX(3),     // ONNX Runtime (CPU or with delegates)
    NCNN(4),     // Tencent NCNN
    LLAMA(5),    // llama.cpp for LLM inference
    CPU(10);     // Pure CPU fallback

    companion object {
        fun fromString(name: String): BackendType? {
            return entries.find { it.name.equals(name, ignoreCase = true) }
        }
    }
}

/**
 * Backend configuration
 */
data class BackendConfig(
    val modelPath: String,
    val inputSize: Int = 256,
    val embeddingDim: Int = 512,
    val numThreads: Int = 4,
    val useGpu: Boolean = false,
    val useFp16: Boolean = false,
    val extraOptions: Map<String, Any> = emptyMap()
)

/**
 * Backend capabilities
 */
data class BackendCapabilities(
    val backendType: BackendType,
    val isAvailable: Boolean,
    val supportsFp16: Boolean = false,
    val supportsInt8: Boolean = false,
    val supportsDynamicShapes: Boolean = false,
    val maxBatchSize: Int = 1,
    val estimatedTflops: Float = 0f,
    val acceleratorName: String = "Unknown",
    val notes: String = ""
)

/**
 * Inference statistics
 */
data class InferenceStats(
    val totalInferences: Long = 0,
    val averageLatencyMs: Float = 0f,
    val minLatencyMs: Float = Float.MAX_VALUE,
    val maxLatencyMs: Float = 0f,
    val lastLatencyMs: Float = 0f,
    val warmupComplete: Boolean = false
) {
    fun withNewSample(latencyMs: Float): InferenceStats {
        val newTotal = totalInferences + 1
        val newAvg = if (totalInferences == 0L) latencyMs
                     else (averageLatencyMs * totalInferences + latencyMs) / newTotal
        return InferenceStats(
            totalInferences = newTotal,
            averageLatencyMs = newAvg,
            minLatencyMs = minOf(minLatencyMs, latencyMs),
            maxLatencyMs = maxOf(maxLatencyMs, latencyMs),
            lastLatencyMs = latencyMs,
            warmupComplete = newTotal >= 3
        )
    }
}

/**
 * Result of backend selection
 */
data class BackendSelection(
    val backend: VisionBackend,
    val reason: String,
    val alternatives: List<BackendType>
)
