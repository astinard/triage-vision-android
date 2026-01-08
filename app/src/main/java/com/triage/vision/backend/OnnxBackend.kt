package com.triage.vision.backend

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * ONNX Runtime backend for vision inference.
 *
 * This backend uses ONNX Runtime with optional NNAPI delegate for
 * hardware acceleration on supported devices.
 *
 * Features:
 * - Automatic NNAPI acceleration when available
 * - CPU fallback for unsupported operations
 * - FP16 support on compatible hardware
 * - Thread pool configuration
 */
class OnnxBackend : VisionBackend {

    companion object {
        private const val TAG = "OnnxBackend"
    }

    override val type = BackendType.ONNX
    override val name = "ONNX Runtime"

    private var ortEnvironment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var isInitialized = false
    private var stats = InferenceStats()

    // Configuration
    private var inputSize = 256
    private var embeddingDim = 512
    private var mean = floatArrayOf(0f, 0f, 0f)
    private var std = floatArrayOf(1f, 1f, 1f)
    private var useNnapi = false

    override suspend fun initialize(context: Context, config: BackendConfig): Boolean =
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Initializing ONNX backend with model: ${config.modelPath}")

                inputSize = config.inputSize
                embeddingDim = config.embeddingDim

                // Get normalization from extra options
                @Suppress("UNCHECKED_CAST")
                (config.extraOptions["mean"] as? FloatArray)?.let { mean = it }
                @Suppress("UNCHECKED_CAST")
                (config.extraOptions["std"] as? FloatArray)?.let { std = it }

                // Create ONNX environment
                ortEnvironment = OrtEnvironment.getEnvironment()

                // Configure session options
                val sessionOptions = OrtSession.SessionOptions().apply {
                    // Set number of threads
                    setIntraOpNumThreads(config.numThreads)

                    // Try to enable NNAPI
                    try {
                        addNnapi()
                        useNnapi = true
                        Log.i(TAG, "NNAPI acceleration enabled")
                    } catch (e: Exception) {
                        useNnapi = false
                        Log.w(TAG, "NNAPI not available: ${e.message}")
                    }

                    // Enable optimizations
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                }

                // Load model from assets or file path
                val modelBytes = if (config.modelPath.startsWith("/")) {
                    java.io.File(config.modelPath).readBytes()
                } else {
                    context.assets.open(config.modelPath).use { it.readBytes() }
                }

                session = ortEnvironment?.createSession(modelBytes, sessionOptions)
                isInitialized = true

                Log.i(TAG, "ONNX backend initialized (NNAPI: $useNnapi)")
                true

            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize ONNX backend: ${e.message}", e)
                isInitialized = false
                false
            }
        }

    override fun isReady(): Boolean = isInitialized

    override fun getCapabilities(): BackendCapabilities {
        return BackendCapabilities(
            backendType = BackendType.ONNX,
            isAvailable = true,
            supportsFp16 = useNnapi,  // NNAPI may support FP16
            supportsInt8 = false,     // Would need quantized model
            supportsDynamicShapes = true,
            maxBatchSize = 1,
            estimatedTflops = if (useNnapi) 1f else 0.1f,
            acceleratorName = if (useNnapi) "NNAPI" else "CPU",
            notes = if (useNnapi) "Using NNAPI delegate" else "CPU-only mode"
        )
    }

    override suspend fun processImage(bitmap: Bitmap): FloatArray? = withContext(Dispatchers.Default) {
        if (!isInitialized || session == null) {
            Log.w(TAG, "ONNX backend not initialized")
            return@withContext null
        }

        val startTime = System.currentTimeMillis()

        try {
            // Preprocess image
            val inputTensor = preprocessImage(bitmap)

            // Run inference
            val inputs = mapOf("input" to inputTensor)
            val outputs = session?.run(inputs)

            // Extract embedding
            val outputTensor = outputs?.get(0) as? OnnxTensor
            val rawOutput = outputTensor?.value

            val embedding = when (rawOutput) {
                is Array<*> -> {
                    @Suppress("UNCHECKED_CAST")
                    (rawOutput as? Array<FloatArray>)?.get(0)
                }
                is FloatArray -> rawOutput
                else -> null
            }

            // Normalize embedding
            val normalizedEmbedding = embedding?.let { normalize(it) }

            // Clean up
            inputTensor.close()
            outputs?.close()

            val elapsed = System.currentTimeMillis() - startTime
            stats = stats.withNewSample(elapsed.toFloat())

            normalizedEmbedding

        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}", e)
            null
        }
    }

    override suspend fun classify(bitmap: Bitmap, labelEmbeddings: Array<FloatArray>): FloatArray? {
        val imageEmbedding = processImage(bitmap) ?: return null

        // Compute cosine similarities (embeddings are already normalized)
        return labelEmbeddings.map { labelEmb ->
            var dot = 0f
            for (i in imageEmbedding.indices) {
                dot += imageEmbedding[i] * labelEmb[i]
            }
            dot
        }.toFloatArray()
    }

    override fun getStats(): InferenceStats = stats

    override fun close() {
        try {
            session?.close()
            ortEnvironment?.close()
            session = null
            ortEnvironment = null
            isInitialized = false
            Log.i(TAG, "ONNX backend closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX backend: ${e.message}")
        }
    }

    // ========== Image Preprocessing ==========

    private fun preprocessImage(bitmap: Bitmap): OnnxTensor {
        // Resize to input size
        val resized = if (bitmap.width != inputSize || bitmap.height != inputSize) {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        } else {
            bitmap
        }

        // Convert to float array with normalization
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // ONNX expects [1, 3, H, W] in CHW format
        val floatBuffer = FloatBuffer.allocate(3 * inputSize * inputSize)

        // Channel-first format (CHW)
        for (c in 0 until 3) {
            for (h in 0 until inputSize) {
                for (w in 0 until inputSize) {
                    val pixel = pixels[h * inputSize + w]
                    val value = when (c) {
                        0 -> ((pixel shr 16) and 0xFF) / 255f  // R
                        1 -> ((pixel shr 8) and 0xFF) / 255f   // G
                        else -> (pixel and 0xFF) / 255f        // B
                    }
                    // Normalize with mean/std
                    floatBuffer.put((value - mean[c]) / std[c])
                }
            }
        }

        floatBuffer.rewind()

        if (resized != bitmap) {
            resized.recycle()
        }

        return OnnxTensor.createTensor(
            ortEnvironment,
            floatBuffer,
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )
    }

    private fun normalize(embedding: FloatArray): FloatArray {
        val norm = sqrt(embedding.map { it * it }.sum())
        return if (norm > 0) {
            embedding.map { it / norm }.toFloatArray()
        } else {
            embedding
        }
    }
}
