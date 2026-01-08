package com.triage.vision.backend

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File

/**
 * Registry for discovering and selecting vision processing backends.
 *
 * The registry automatically detects available backends based on:
 * - Device hardware (chipset, NPU availability)
 * - Installed libraries (QNN SDK, NNAPI version)
 * - Runtime capabilities
 *
 * Selection priority: QNN > NNAPI > ONNX > NCNN > CPU
 */
object BackendRegistry {

    private const val TAG = "BackendRegistry"

    // Registered backend factories
    private val factories = mutableMapOf<BackendType, () -> VisionBackend>()

    // Cached capabilities
    private var deviceCapabilities: DeviceCapabilities? = null

    /**
     * Register a backend factory
     */
    fun register(type: BackendType, factory: () -> VisionBackend) {
        factories[type] = factory
        Log.d(TAG, "Registered backend: $type")
    }

    /**
     * Get all registered backend types
     */
    fun getRegisteredTypes(): Set<BackendType> = factories.keys.toSet()

    /**
     * Detect device capabilities
     */
    fun detectCapabilities(context: Context): DeviceCapabilities {
        deviceCapabilities?.let { return it }

        val caps = DeviceCapabilities(
            chipset = detectChipset(),
            hasQnn = detectQnnAvailability(context),
            hasNnapi = detectNnapiAvailability(),
            nnapiVersion = getNnapiVersion(),
            hasNpu = detectNpuAvailability(),
            npuTops = estimateNpuTops(),
            deviceModel = Build.MODEL,
            androidVersion = Build.VERSION.SDK_INT
        )

        deviceCapabilities = caps
        Log.i(TAG, "Device capabilities detected: $caps")
        return caps
    }

    /**
     * Select the best available backend for image classification
     *
     * @param context Android context
     * @param preferredType Optional preferred backend type
     * @return Selected backend with selection reason
     */
    fun selectBackend(
        context: Context,
        preferredType: BackendType? = null
    ): BackendSelection? {
        val caps = detectCapabilities(context)

        // If preferred type is specified and available, use it
        if (preferredType != null && factories.containsKey(preferredType)) {
            val backend = factories[preferredType]?.invoke()
            if (backend != null) {
                return BackendSelection(
                    backend = backend,
                    reason = "User preferred: ${preferredType.name}",
                    alternatives = getAlternatives(preferredType)
                )
            }
        }

        // Auto-select based on capabilities and priority
        val candidates = factories.keys
            .sortedBy { it.priority }
            .filter { isBackendSuitable(it, caps) }

        for (type in candidates) {
            val backend = factories[type]?.invoke()
            if (backend != null) {
                val reason = getSelectionReason(type, caps)
                return BackendSelection(
                    backend = backend,
                    reason = reason,
                    alternatives = candidates.filter { it != type }
                )
            }
        }

        Log.e(TAG, "No suitable backend found")
        return null
    }

    /**
     * Create a specific backend type
     */
    fun createBackend(type: BackendType): VisionBackend? {
        return factories[type]?.invoke()
    }

    /**
     * Check if a backend type is suitable for the device
     */
    private fun isBackendSuitable(type: BackendType, caps: DeviceCapabilities): Boolean {
        return when (type) {
            BackendType.QNN -> caps.hasQnn && caps.hasNpu
            BackendType.NNAPI -> caps.hasNnapi && caps.nnapiVersion >= 29
            BackendType.ONNX -> true  // Always available as fallback
            BackendType.NCNN -> true  // Always available
            BackendType.LLAMA -> true // Always available
            BackendType.CPU -> true   // Always available
        }
    }

    /**
     * Get reason for selecting a backend
     */
    private fun getSelectionReason(type: BackendType, caps: DeviceCapabilities): String {
        return when (type) {
            BackendType.QNN -> "QNN selected: ${caps.chipset} with ${caps.npuTops} TOPS NPU"
            BackendType.NNAPI -> "NNAPI v${caps.nnapiVersion} selected for hardware acceleration"
            BackendType.ONNX -> "ONNX Runtime selected (NNAPI delegate if available)"
            BackendType.NCNN -> "NCNN selected for optimized ARM inference"
            BackendType.LLAMA -> "llama.cpp selected for LLM inference"
            BackendType.CPU -> "CPU fallback selected (no accelerator available)"
        }
    }

    /**
     * Get alternative backends
     */
    private fun getAlternatives(selected: BackendType): List<BackendType> {
        return factories.keys
            .filter { it != selected }
            .sortedBy { it.priority }
    }

    // ========== Device Detection ==========

    /**
     * Detect the device chipset using Build properties
     */
    private fun detectChipset(): String {
        // Use Build properties instead of shell commands for safety
        val board = Build.BOARD.lowercase()
        val hardware = Build.HARDWARE.lowercase()
        // Build.SOC_MODEL requires API 31+
        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.lowercase()
        } else {
            ""
        }

        // Map common chipsets based on Build properties
        return when {
            // Qualcomm QCM6490 (Mason Scan 600 target)
            soc.contains("qcm6490") || board.contains("qcm6490") -> "QCM6490"

            // Snapdragon 8 series
            soc.contains("sm8550") || hardware.contains("sm8550") -> "Snapdragon 8 Gen 2"
            soc.contains("sm8450") || hardware.contains("sm8450") -> "Snapdragon 8 Gen 1"
            soc.contains("sm8350") || hardware.contains("sm8350") -> "Snapdragon 888"

            // Older Snapdragon
            soc.contains("sdm845") || hardware.contains("sdm845") -> "Snapdragon 845"
            soc.contains("msm8998") || hardware.contains("msm8998") -> "Snapdragon 835"
            board.contains("walleye") || board.contains("taimen") -> "Snapdragon 835"  // Pixel 2
            board.contains("blueline") || board.contains("crosshatch") -> "Snapdragon 845"  // Pixel 3

            // Google Tensor
            soc.contains("tensor") || hardware.contains("tensor") -> "Google Tensor"
            board.contains("oriole") || board.contains("raven") -> "Google Tensor"  // Pixel 6
            board.contains("cheetah") || board.contains("panther") -> "Google Tensor G2"  // Pixel 7

            // Samsung Exynos
            soc.contains("exynos") || hardware.contains("exynos") -> "Exynos"

            // Fallback to hardware name
            hardware.isNotEmpty() -> hardware
            board.isNotEmpty() -> board
            else -> "unknown"
        }
    }

    /**
     * Check if QNN SDK is available
     */
    private fun detectQnnAvailability(context: Context): Boolean {
        // Check for QNN shared libraries
        val qnnLibs = listOf(
            "libQnnHtp.so",
            "libQnnSystem.so",
            "libQnnHtpPrepare.so"
        )

        // Check system lib paths
        val libPaths = listOf(
            "/vendor/lib64",
            "/system/lib64",
            "/system/vendor/lib64"
        )

        for (lib in qnnLibs) {
            for (path in libPaths) {
                if (File("$path/$lib").exists()) {
                    Log.d(TAG, "Found QNN library: $path/$lib")
                    return true
                }
            }
        }

        // Also check if we've bundled QNN libs
        val nativeLibDir = context.applicationInfo.nativeLibraryDir
        for (lib in qnnLibs) {
            if (File("$nativeLibDir/$lib").exists()) {
                Log.d(TAG, "Found bundled QNN library: $nativeLibDir/$lib")
                return true
            }
        }

        return false
    }

    /**
     * Check NNAPI availability
     */
    private fun detectNnapiAvailability(): Boolean {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.P  // NNAPI 1.1+ from Android 9
    }

    /**
     * Get NNAPI version
     */
    private fun getNnapiVersion(): Int {
        return when {
            Build.VERSION.SDK_INT >= 34 -> 34  // Android 14
            Build.VERSION.SDK_INT >= 33 -> 33  // Android 13
            Build.VERSION.SDK_INT >= 31 -> 31  // Android 12
            Build.VERSION.SDK_INT >= 30 -> 30  // Android 11
            Build.VERSION.SDK_INT >= 29 -> 29  // Android 10
            Build.VERSION.SDK_INT >= 28 -> 28  // Android 9
            else -> 0
        }
    }

    /**
     * Check NPU availability based on known chipsets
     */
    private fun detectNpuAvailability(): Boolean {
        val chipset = detectChipset()
        return when {
            // Qualcomm with Hexagon DSP/NPU
            chipset.contains("QCM6490") -> true
            chipset.contains("Snapdragon 8") -> true
            chipset.contains("Snapdragon 7") -> true
            chipset.contains("845") -> true  // Has Hexagon 685
            chipset.contains("855") -> true
            chipset.contains("865") -> true
            chipset.contains("888") -> true
            // Google Tensor
            chipset.contains("Tensor") -> true
            // Samsung Exynos with NPU
            chipset.contains("Exynos 2") -> true
            else -> false
        }
    }

    /**
     * Estimate NPU performance in TOPS (Tera Operations Per Second)
     */
    private fun estimateNpuTops(): Float {
        val chipset = detectChipset()
        return when {
            chipset.contains("QCM6490") -> 10f    // Target device
            chipset.contains("8 Gen 2") -> 45f
            chipset.contains("8 Gen 1") -> 27f
            chipset.contains("888") -> 26f
            chipset.contains("865") -> 15f
            chipset.contains("855") -> 7f
            chipset.contains("845") -> 3f
            chipset.contains("835") -> 1.5f
            chipset.contains("Tensor G3") -> 12f
            chipset.contains("Tensor G2") -> 8f
            chipset.contains("Tensor") -> 5f
            else -> 0f
        }
    }

    /**
     * Check if running on the target Mason Scan 600 device
     */
    fun isMasonScan600(): Boolean {
        val chipset = detectChipset()
        val model = Build.MODEL.lowercase()
        return chipset.contains("QCM6490", ignoreCase = true) ||
               model.contains("mason") ||
               model.contains("scan 600")
    }

    /**
     * Clear cached capabilities (for testing)
     */
    fun clearCache() {
        deviceCapabilities = null
    }
}

/**
 * Device capabilities data class
 */
data class DeviceCapabilities(
    val chipset: String,
    val hasQnn: Boolean,
    val hasNnapi: Boolean,
    val nnapiVersion: Int,
    val hasNpu: Boolean,
    val npuTops: Float,
    val deviceModel: String,
    val androidVersion: Int
) {
    fun getSummary(): String {
        return buildString {
            append("$deviceModel ($chipset)")
            if (hasNpu) append(" - NPU: ${npuTops}T")
            if (hasQnn) append(" [QNN]")
            if (hasNnapi) append(" [NNAPI v$nnapiVersion]")
        }
    }
}
