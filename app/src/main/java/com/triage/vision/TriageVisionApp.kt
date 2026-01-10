package com.triage.vision

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
import com.triage.vision.backend.BackendRegistry
import com.triage.vision.backend.BackendType
import com.triage.vision.backend.OnnxBackend
import com.triage.vision.backend.QnnBackend
import com.triage.vision.camera.MediaPipePoseDetector
import com.triage.vision.classifier.FastClassifier
import com.triage.vision.data.AppDatabase
import com.triage.vision.data.ObservationRepository
import com.triage.vision.native.NativeBridge
import com.triage.vision.pipeline.FastPipeline
import com.triage.vision.pipeline.SlowPipeline
import com.triage.vision.charting.AutoChartEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File

/**
 * Application class for Triage Vision
 *
 * Initializes:
 * - Native ML libraries (NCNN, llama.cpp)
 * - Database
 * - Notification channels
 */
class TriageVisionApp : Application() {

    companion object {
        private const val TAG = "TriageVisionApp"

        const val CHANNEL_MONITORING = "monitoring"
        const val CHANNEL_ALERTS = "alerts"

        lateinit var instance: TriageVisionApp
            private set
    }

    // Application-scoped coroutine scope
    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // Core components (lazy initialized)
    val nativeBridge: NativeBridge by lazy { NativeBridge() }
    // MediaPipe pose detector - disabled on Pixel 2 due to native crashes
    // Enable only on devices with known MediaPipe compatibility (Mason Scan 600, Pixel 4+, etc.)
    val poseDetector: MediaPipePoseDetector? by lazy {
        if (isMediaPipeCompatible()) {
            try {
                MediaPipePoseDetector(this)
            } catch (e: Exception) {
                Log.w(TAG, "MediaPipe initialization failed: ${e.message}")
                null
            }
        } else {
            Log.i(TAG, "MediaPipe disabled on ${android.os.Build.MODEL} (known incompatible)")
            null
        }
    }
    val fastPipeline: FastPipeline by lazy { FastPipeline(nativeBridge, poseDetector) }

    /**
     * Check if MediaPipe is compatible with this device
     * MediaPipe crashes on Pixel 2 and some older devices
     */
    private fun isMediaPipeCompatible(): Boolean {
        val model = android.os.Build.MODEL.lowercase()
        val sdk = android.os.Build.VERSION.SDK_INT

        // Known incompatible devices
        val incompatible = listOf(
            "pixel 2",      // Crashes in native code
            "pixel 2 xl",   // Same issue
            "pixel",        // Original Pixel may have issues
            "pixel xl"      // Original Pixel XL
        )

        if (incompatible.any { model.contains(it) }) {
            return false
        }

        // Require API 29+ for stable MediaPipe (Android 10+)
        // But for testing on older devices, we allow it if not in blocklist
        return sdk >= 29 || BackendRegistry.isMasonScan600()
    }
    val slowPipeline: SlowPipeline by lazy { SlowPipeline(nativeBridge) }
    val chartEngine: AutoChartEngine by lazy { AutoChartEngine() }

    // CLIP-based fast classifier for real-time position/alertness detection
    val fastClassifier: FastClassifier by lazy { FastClassifier(this) }

    // Database
    val database: AppDatabase by lazy { AppDatabase.getInstance(this) }
    val observationRepository: ObservationRepository by lazy {
        ObservationRepository(database.observationDao())
    }

    // Initialization state
    var isNativeInitialized = false
        private set

    override fun onCreate() {
        super.onCreate()
        instance = this

        Log.i(TAG, "Triage Vision App starting...")

        // Register vision backends
        registerBackends()

        // Detect and log device capabilities
        val capabilities = BackendRegistry.detectCapabilities(this)
        Log.i(TAG, "Device: ${capabilities.getSummary()}")
        if (BackendRegistry.isMasonScan600()) {
            Log.i(TAG, "Running on Mason Scan 600 target device!")
        }

        // Create notification channels
        createNotificationChannels()

        // Initialize native libraries in background
        applicationScope.launch {
            initializeNativeLibraries()
        }

        // Initialize database
        applicationScope.launch(Dispatchers.IO) {
            // Trigger database creation
            database.observationDao()
            Log.i(TAG, "Database initialized")
        }
    }

    /**
     * Register available vision processing backends
     */
    private fun registerBackends() {
        // Register QNN backend (for Mason Scan 600 with QCM6490)
        BackendRegistry.register(BackendType.QNN) { QnnBackend() }

        // Register ONNX Runtime backend (default for most devices)
        BackendRegistry.register(BackendType.ONNX) { OnnxBackend() }

        Log.i(TAG, "Registered backends: ${BackendRegistry.getRegisteredTypes()}")
    }

    private fun createNotificationChannels() {
        val notificationManager = getSystemService(NotificationManager::class.java)

        // Monitoring channel (low importance - ongoing)
        val monitoringChannel = NotificationChannel(
            CHANNEL_MONITORING,
            getString(R.string.notification_channel_monitoring),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Ongoing patient monitoring notifications"
            setShowBadge(false)
        }

        // Alerts channel (high importance)
        val alertsChannel = NotificationChannel(
            CHANNEL_ALERTS,
            getString(R.string.notification_channel_alerts),
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            description = "Critical patient monitoring alerts"
            enableVibration(true)
            vibrationPattern = longArrayOf(0, 500, 200, 500)
        }

        notificationManager.createNotificationChannels(
            listOf(monitoringChannel, alertsChannel)
        )

        Log.i(TAG, "Notification channels created")
    }

    private suspend fun initializeNativeLibraries() {
        try {
            // Get model directory path
            val modelDir = getModelDirectory()

            Log.i(TAG, "Initializing native libraries from: $modelDir")

            // Check if models exist (match actual asset filenames)
            val yoloModel = File(modelDir, "yolo11n_ncnn_model/model.ncnn.bin")
            val vlmModel = File(modelDir, "SmolVLM-500M-Instruct-Q8_0.gguf")

            if (!yoloModel.exists()) {
                Log.w(TAG, "YOLO model not found at: ${yoloModel.absolutePath}")
                Log.w(TAG, "Fast pipeline will not be available")
            } else {
                Log.i(TAG, "YOLO model found: ${yoloModel.absolutePath} (${yoloModel.length() / 1024}KB)")
            }

            val mmproj = File(modelDir, "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf")

            if (!vlmModel.exists()) {
                Log.w(TAG, "SmolVLM model not found at: ${vlmModel.absolutePath}")
                Log.w(TAG, "Slow pipeline will not be available")
            } else {
                Log.i(TAG, "VLM model found: ${vlmModel.absolutePath} (${vlmModel.length() / 1024 / 1024}MB)")
            }

            if (!mmproj.exists()) {
                Log.w(TAG, "VLM mmproj not found at: ${mmproj.absolutePath}")
                Log.w(TAG, "VLM vision capabilities will not be available")
            } else {
                Log.i(TAG, "VLM mmproj found: ${mmproj.absolutePath} (${mmproj.length() / 1024 / 1024}MB)")
            }

            // Initialize native bridge
            val result = nativeBridge.init(modelDir)

            if (result == 0) {
                isNativeInitialized = true
                Log.i(TAG, "Native libraries initialized successfully")
            } else {
                Log.e(TAG, "Native library initialization failed with code: $result")
            }

        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native library: ${e.message}")
            Log.e(TAG, "Make sure libncnn.so and libtriage_vision.so are built for arm64-v8a")
        } catch (e: Exception) {
            Log.e(TAG, "Native initialization error: ${e.message}", e)
        }
    }

    /**
     * Get the directory where ML models are stored
     */
    fun getModelDirectory(): String {
        // First check external files (for easier model updates)
        val externalDir = getExternalFilesDir("models")
        if (externalDir?.exists() == true) {
            val files = externalDir.listFiles()
            if (files?.isNotEmpty() == true) {
                return externalDir.absolutePath
            }
        }

        // Fall back to internal files
        val internalDir = File(filesDir, "models")
        if (!internalDir.exists()) {
            internalDir.mkdirs()
        }

        // Copy from assets if needed
        copyModelsFromAssets(internalDir)

        return internalDir.absolutePath
    }

    private fun copyModelsFromAssets(targetDir: File) {
        try {
            copyAssetDirectory("models", targetDir)
        } catch (e: Exception) {
            Log.w(TAG, "Could not copy models from assets: ${e.message}")
        }
    }

    private fun copyAssetDirectory(assetPath: String, targetDir: File) {
        val assetManager = assets
        val entries = assetManager.list(assetPath) ?: return

        if (!targetDir.exists()) {
            targetDir.mkdirs()
        }

        for (entry in entries) {
            val assetEntryPath = "$assetPath/$entry"
            val targetFile = File(targetDir, entry)

            // Check if this is a directory by trying to list its contents
            val subEntries = assetManager.list(assetEntryPath)
            if (subEntries != null && subEntries.isNotEmpty()) {
                // It's a directory, recurse
                copyAssetDirectory(assetEntryPath, targetFile)
            } else {
                // It's a file
                if (!targetFile.exists()) {
                    Log.i(TAG, "Copying model from assets: $assetEntryPath")
                    assetManager.open(assetEntryPath).use { input ->
                        targetFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
            }
        }
    }

    override fun onTerminate() {
        super.onTerminate()
        poseDetector?.close()
        nativeBridge.cleanup()
    }
}
