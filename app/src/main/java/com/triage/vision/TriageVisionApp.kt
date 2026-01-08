package com.triage.vision

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.util.Log
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
    // MediaPipe crashes on older devices (Pixel 2, etc.) - disable for now
    // TODO: Enable on newer devices with compatible GPU (API 28+, Adreno 6xx+)
    // val poseDetector: MediaPipePoseDetector by lazy { MediaPipePoseDetector(this) }
    val fastPipeline: FastPipeline by lazy { FastPipeline(nativeBridge, null) }
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
        // poseDetector.close()  // Disabled - MediaPipe crashes on older devices
        nativeBridge.cleanup()
    }
}
