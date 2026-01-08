package com.triage.vision.backend

import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for BackendRegistry.
 *
 * Tests the backend registration and selection logic.
 * Note: Device detection tests require Android instrumented tests.
 */
class BackendRegistryTest {

    @Before
    fun setUp() {
        // Clear any previously registered backends
        BackendRegistry.clearCache()
    }

    @After
    fun tearDown() {
        BackendRegistry.clearCache()
    }

    // ==================== Backend Type Tests ====================

    @Test
    fun `backend types have correct priority order`() {
        // QNN should have highest priority (lowest number)
        assertTrue(BackendType.QNN.priority < BackendType.NNAPI.priority)
        assertTrue(BackendType.NNAPI.priority < BackendType.ONNX.priority)
        assertTrue(BackendType.ONNX.priority < BackendType.NCNN.priority)
        assertTrue(BackendType.NCNN.priority < BackendType.LLAMA.priority)
        assertTrue(BackendType.LLAMA.priority < BackendType.CPU.priority)
    }

    @Test
    fun `backend types are all unique`() {
        val types = BackendType.entries
        val priorities = types.map { it.priority }
        assertEquals("All backend types should have unique priorities",
            priorities.size, priorities.toSet().size)
    }

    // ==================== Capabilities Tests ====================

    @Test
    fun `backend capabilities reports correctly`() {
        val caps = BackendCapabilities(
            backendType = BackendType.ONNX,
            isAvailable = true,
            supportsFp16 = true,
            supportsInt8 = false,
            supportsDynamicShapes = true,
            maxBatchSize = 4,
            estimatedTflops = 2.5f,
            acceleratorName = "NNAPI",
            notes = "Test backend"
        )

        assertEquals(BackendType.ONNX, caps.backendType)
        assertTrue(caps.isAvailable)
        assertTrue(caps.supportsFp16)
        assertFalse(caps.supportsInt8)
        assertTrue(caps.supportsDynamicShapes)
        assertEquals(4, caps.maxBatchSize)
        assertEquals(2.5f, caps.estimatedTflops, 0.01f)
        assertEquals("NNAPI", caps.acceleratorName)
    }

    // ==================== Backend Config Tests ====================

    @Test
    fun `backend config default values`() {
        val config = BackendConfig(modelPath = "models/test.onnx")

        assertEquals("models/test.onnx", config.modelPath)
        assertEquals(256, config.inputSize)  // Default
        assertEquals(512, config.embeddingDim)  // Default
        assertEquals(4, config.numThreads)  // Default
        assertFalse(config.useGpu)  // Default
        assertFalse(config.useFp16)  // Default
        assertTrue(config.extraOptions.isEmpty())
    }

    @Test
    fun `backend config custom values`() {
        val config = BackendConfig(
            modelPath = "models/custom.onnx",
            inputSize = 224,
            embeddingDim = 768,
            numThreads = 2,
            useFp16 = false,
            extraOptions = mapOf("key" to "value")
        )

        assertEquals("models/custom.onnx", config.modelPath)
        assertEquals(224, config.inputSize)
        assertEquals(768, config.embeddingDim)
        assertEquals(2, config.numThreads)
        assertFalse(config.useFp16)
        assertEquals("value", config.extraOptions["key"])
    }

    // ==================== Inference Stats Tests ====================

    @Test
    fun `inference stats initial values`() {
        val stats = InferenceStats()

        assertEquals(0L, stats.totalInferences)
        assertEquals(0f, stats.averageLatencyMs, 0.01f)
        assertEquals(Float.MAX_VALUE, stats.minLatencyMs, 0.01f)
        assertEquals(0f, stats.maxLatencyMs, 0.01f)
        assertEquals(0f, stats.lastLatencyMs, 0.01f)
        assertFalse(stats.warmupComplete)
    }

    @Test
    fun `inference stats withNewSample updates correctly`() {
        val stats = InferenceStats()

        val updated = stats.withNewSample(100f)

        assertEquals(1L, updated.totalInferences)
        assertEquals(100f, updated.averageLatencyMs, 0.01f)
        assertEquals(100f, updated.minLatencyMs, 0.01f)
        assertEquals(100f, updated.maxLatencyMs, 0.01f)
        assertEquals(100f, updated.lastLatencyMs, 0.01f)
        assertFalse(updated.warmupComplete) // Need 3 samples
    }

    @Test
    fun `inference stats accumulates multiple samples`() {
        var stats = InferenceStats()

        stats = stats.withNewSample(50f)
        stats = stats.withNewSample(100f)
        stats = stats.withNewSample(150f)

        assertEquals(3L, stats.totalInferences)
        assertEquals(100f, stats.averageLatencyMs, 0.01f)
        assertEquals(50f, stats.minLatencyMs, 0.01f)
        assertEquals(150f, stats.maxLatencyMs, 0.01f)
        assertEquals(150f, stats.lastLatencyMs, 0.01f)
        assertTrue(stats.warmupComplete) // After 3 samples
    }

    @Test
    fun `inference stats immutability`() {
        val original = InferenceStats()
        val updated = original.withNewSample(100f)

        // Original should be unchanged
        assertEquals(0L, original.totalInferences)
        assertEquals(1L, updated.totalInferences)
    }

    // ==================== Device Capabilities Tests ====================

    @Test
    fun `device capabilities getSummary formats correctly`() {
        val caps = DeviceCapabilities(
            chipset = "QCM6490",
            hasQnn = true,
            hasNnapi = true,
            nnapiVersion = 33,
            hasNpu = true,
            npuTops = 10f,
            deviceModel = "Mason Scan 600",
            androidVersion = 33
        )

        val summary = caps.getSummary()

        assertTrue(summary.contains("Mason Scan 600"))
        assertTrue(summary.contains("QCM6490"))
        assertTrue(summary.contains("10.0T"))
        assertTrue(summary.contains("[QNN]"))
        assertTrue(summary.contains("[NNAPI v33]"))
    }

    @Test
    fun `device capabilities without NPU`() {
        val caps = DeviceCapabilities(
            chipset = "Generic",
            hasQnn = false,
            hasNnapi = true,
            nnapiVersion = 29,
            hasNpu = false,
            npuTops = 0f,
            deviceModel = "Test Device",
            androidVersion = 29
        )

        val summary = caps.getSummary()

        assertTrue(summary.contains("Test Device"))
        assertFalse(summary.contains("[QNN]"))
        assertTrue(summary.contains("[NNAPI v29]"))
        assertFalse(summary.contains("NPU:"))  // No NPU info shown when no NPU
    }

    // ==================== Backend Selection Tests ====================

    @Test
    fun `backend selection includes reason and alternatives`() {
        // Create a mock backend selection
        val mockBackend = object : VisionBackend {
            override val type = BackendType.ONNX
            override val name = "Test ONNX Backend"
            override suspend fun initialize(context: android.content.Context, config: BackendConfig) = true
            override fun isReady() = true
            override fun getCapabilities() = BackendCapabilities(
                backendType = BackendType.ONNX,
                isAvailable = true
            )
            override suspend fun processImage(bitmap: android.graphics.Bitmap): FloatArray? = null
            override suspend fun classify(bitmap: android.graphics.Bitmap, labelEmbeddings: Array<FloatArray>): FloatArray? = null
            override fun getStats() = InferenceStats()
            override fun close() {}
        }

        val selection = BackendSelection(
            backend = mockBackend,
            reason = "Test selection",
            alternatives = listOf(BackendType.CPU)
        )

        assertEquals(mockBackend, selection.backend)
        assertEquals("Test selection", selection.reason)
        assertEquals(1, selection.alternatives.size)
        assertEquals(BackendType.CPU, selection.alternatives[0])
    }

    // ==================== Registration Tests ====================

    @Test
    fun `registered types initially empty then populated`() {
        // After registration
        BackendRegistry.register(BackendType.ONNX) { createMockBackend(BackendType.ONNX) }

        val types = BackendRegistry.getRegisteredTypes()
        assertTrue(types.contains(BackendType.ONNX))
    }

    @Test
    fun `multiple backends can be registered`() {
        BackendRegistry.register(BackendType.ONNX) { createMockBackend(BackendType.ONNX) }
        BackendRegistry.register(BackendType.QNN) { createMockBackend(BackendType.QNN) }

        val types = BackendRegistry.getRegisteredTypes()
        assertTrue(types.contains(BackendType.ONNX))
        assertTrue(types.contains(BackendType.QNN))
        assertEquals(2, types.size)
    }

    @Test
    fun `createBackend returns correct type`() {
        BackendRegistry.register(BackendType.ONNX) { createMockBackend(BackendType.ONNX) }

        val backend = BackendRegistry.createBackend(BackendType.ONNX)

        assertNotNull(backend)
        assertEquals(BackendType.ONNX, backend?.type)
    }

    @Test
    fun `createBackend returns null for unregistered type`() {
        // Don't register anything

        val backend = BackendRegistry.createBackend(BackendType.QNN)

        assertNull(backend)
    }

    // ==================== Helper Methods ====================

    private fun createMockBackend(type: BackendType): VisionBackend {
        return object : VisionBackend {
            override val type = type
            override val name = "Mock $type Backend"
            override suspend fun initialize(context: android.content.Context, config: BackendConfig) = true
            override fun isReady() = true
            override fun getCapabilities() = BackendCapabilities(
                backendType = type,
                isAvailable = true
            )
            override suspend fun processImage(bitmap: android.graphics.Bitmap): FloatArray? = null
            override suspend fun classify(bitmap: android.graphics.Bitmap, labelEmbeddings: Array<FloatArray>): FloatArray? = null
            override fun getStats() = InferenceStats()
            override fun close() {}
        }
    }
}
