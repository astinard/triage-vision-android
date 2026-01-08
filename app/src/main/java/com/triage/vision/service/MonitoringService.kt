package com.triage.vision.service

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Binder
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.triage.vision.R
import com.triage.vision.TriageVisionApp
import com.triage.vision.data.AlertEntity
import com.triage.vision.data.ObservationEntity
import com.triage.vision.classifier.FastClassifier
import com.triage.vision.classifier.NursingLabels
import com.triage.vision.pipeline.FastPipeline
import com.triage.vision.pipeline.SlowPipeline
import com.triage.vision.ui.MainActivity
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Foreground service for continuous patient monitoring.
 *
 * Features:
 * - Independent camera processing (works when app in background)
 * - Fast pipeline: YOLO11n motion/pose detection at 15-30 FPS
 * - Slow pipeline: SmolVLM analysis every 5 seconds or on alerts
 * - Fall detection with immediate notification
 * - Stillness alerts after configurable threshold
 * - Wake lock to prevent CPU sleep during monitoring
 * - Binder interface for UI communication
 */
class MonitoringService : Service(), LifecycleOwner {

    companion object {
        private const val TAG = "MonitoringService"
        private const val NOTIFICATION_ID = 1001
        private const val ALERT_NOTIFICATION_ID = 1002

        const val ACTION_START = "com.triage.vision.START_MONITORING"
        const val ACTION_STOP = "com.triage.vision.STOP_MONITORING"
        const val ACTION_TRIGGER_VLM = "com.triage.vision.TRIGGER_VLM"

        // Broadcast actions for UI updates
        const val BROADCAST_STATE_UPDATE = "com.triage.vision.STATE_UPDATE"
        const val BROADCAST_ALERT = "com.triage.vision.ALERT"

        // Configuration
        private const val VLM_INTERVAL_MS = 2 * 1000L // 2 seconds
        private const val WAKE_LOCK_TIMEOUT = 24 * 60 * 60 * 1000L // 24 hours max
    }

    // Lifecycle for CameraX
    private lateinit var lifecycleRegistry: LifecycleRegistry
    override val lifecycle: Lifecycle get() = lifecycleRegistry

    // Core components
    private lateinit var app: TriageVisionApp
    private lateinit var fastPipeline: FastPipeline
    private lateinit var slowPipeline: SlowPipeline
    private lateinit var fastClassifier: FastClassifier

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var cameraExecutor: ExecutorService
    private var imageAnalysis: ImageAnalysis? = null

    // Coroutines
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var vlmSchedulerJob: Job? = null
    private var alertCollectorJob: Job? = null

    // Wake lock
    private var wakeLock: PowerManager.WakeLock? = null

    // State
    private var isMonitoring = false
    private var currentPatientId: String? = null
    private var lastFrame: Bitmap? = null
    private var lastFrameTimestamp = 0L

    // Monitoring state (exposed to UI via binder)
    data class MonitoringState(
        val isMonitoring: Boolean = false,
        val isAnalyzing: Boolean = false,
        val personDetected: Boolean = false,
        val currentPose: FastPipeline.Pose = FastPipeline.Pose.UNKNOWN,
        val poseConfidence: Float = 0f,
        val motionLevel: Float = 0f,
        val secondsSinceMotion: Long = 0,
        val lastObservation: SlowPipeline.Observation? = null,
        val frameCount: Long = 0,
        val fps: Float = 0f,
        val landmarks: List<FastPipeline.PoseLandmark> = emptyList(),
        // CLIP classification results
        val clipPosition: NursingLabels.Position = NursingLabels.Position.LYING_SUPINE,
        val clipPositionConfidence: Float = 0f,
        val clipAlertness: NursingLabels.Alertness = NursingLabels.Alertness.EYES_CLOSED,
        val clipAlertnessConfidence: Float = 0f,
        val clipActivity: NursingLabels.Activity = NursingLabels.Activity.STILL,
        val clipActivityConfidence: Float = 0f,
        val clipSafety: NursingLabels.SafetyConcern = NursingLabels.SafetyConcern.NONE,
        val clipSafetyConfidence: Float = 0f,
        val clipInferenceMs: Long = 0,
        val clipEnabled: Boolean = false
    )

    private val _monitoringState = MutableStateFlow(MonitoringState())
    val monitoringState: StateFlow<MonitoringState> = _monitoringState.asStateFlow()

    private val _alerts = MutableSharedFlow<FastPipeline.Alert>(extraBufferCapacity = 10)
    val alerts: SharedFlow<FastPipeline.Alert> = _alerts.asSharedFlow()

    // FPS tracking
    private var frameCountForFps = 0
    private var fpsUpdateTime = 0L

    // Binder for UI connection
    private val binder = MonitoringBinder()
    private var uiBound = false  // True when UI is actively bound and can feed frames

    inner class MonitoringBinder : Binder() {
        fun getService(): MonitoringService = this@MonitoringService
    }

    /**
     * Called when UI binds - service will receive frames from UI instead of using own camera
     */
    fun onUiBound() {
        Log.i(TAG, "UI bound - will receive frames from UI")
        uiBound = true
        // Stop our own camera if running - UI will feed frames
        if (isMonitoring && cameraProvider != null) {
            Log.i(TAG, "Releasing camera for UI")
            stopCamera()
        }
    }

    /**
     * Called when UI unbinds - service will use own camera
     */
    fun onUiUnbound() {
        Log.i(TAG, "UI unbound - will use own camera")
        uiBound = false
        // Start our own camera if monitoring
        if (isMonitoring && cameraProvider == null) {
            Log.i(TAG, "Starting own camera (UI disconnected)")
            startCamera()
        }
    }

    /**
     * Receive frame from UI for processing
     */
    fun processExternalFrame(bitmap: Bitmap) {
        if (!isMonitoring) {
            // Log occasionally to help debug startup timing
            if (System.currentTimeMillis() - lastFrameTimestamp > 1000) {
                Log.d(TAG, "processExternalFrame called but not monitoring yet")
            }
            return
        }

        // Store for VLM
        lastFrame = bitmap
        lastFrameTimestamp = System.currentTimeMillis()

        // Log first frame received
        if (_monitoringState.value.frameCount == 0L) {
            Log.i(TAG, "First frame received from UI: ${bitmap.width}x${bitmap.height}")
        }

        // Process through fast pipeline
        serviceScope.launch {
            processFrameInternal(bitmap)
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "MonitoringService created")

        lifecycleRegistry = LifecycleRegistry(this)
        lifecycleRegistry.currentState = Lifecycle.State.CREATED

        app = TriageVisionApp.instance
        fastPipeline = app.fastPipeline
        slowPipeline = app.slowPipeline
        fastClassifier = app.fastClassifier

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize CLIP classifier in background
        serviceScope.launch {
            val success = fastClassifier.initialize()
            Log.i(TAG, "FastClassifier initialized: $success (using real model: ${fastClassifier.isUsingRealModel()})")
            _monitoringState.update { it.copy(clipEnabled = success) }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                currentPatientId = intent.getStringExtra("patient_id")
                startMonitoring()
            }
            ACTION_STOP -> stopMonitoring()
            ACTION_TRIGGER_VLM -> {
                serviceScope.launch {
                    runVlmAnalysis("manual")
                }
            }
        }
        return START_STICKY
    }

    private fun startMonitoring() {
        if (isMonitoring) {
            Log.w(TAG, "Already monitoring")
            return
        }

        Log.i(TAG, "Starting foreground monitoring")

        // Acquire wake lock
        acquireWakeLock()

        // Start foreground with notification
        val notification = createMonitoringNotification("Starting...")
        startForeground(NOTIFICATION_ID, notification)

        // Check native initialization
        if (!app.isNativeInitialized) {
            Log.e(TAG, "Native libraries not initialized")
            updateNotification("Error: Native libraries not ready")
            return
        }

        isMonitoring = true
        lifecycleRegistry.currentState = Lifecycle.State.STARTED

        // Start camera
        startCamera()

        // Start VLM scheduler
        startVlmScheduler()

        // Collect fast pipeline alerts
        collectAlerts()

        _monitoringState.update { it.copy(isMonitoring = true) }
        updateNotification("Monitoring patient")

        Log.i(TAG, "Monitoring started successfully")
    }

    private fun stopMonitoring() {
        Log.i(TAG, "Stopping monitoring service")

        isMonitoring = false
        _monitoringState.update { it.copy(isMonitoring = false) }

        // Cancel jobs
        vlmSchedulerJob?.cancel()
        alertCollectorJob?.cancel()

        // Stop camera
        stopCamera()

        // Release wake lock
        releaseWakeLock()

        lifecycleRegistry.currentState = Lifecycle.State.CREATED

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun startCamera() {
        // Skip if UI is bound - UI will feed frames
        if (uiBound) {
            Log.i(TAG, "Skipping camera start - UI is bound and will feed frames")
            return
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()

                // Image analysis for frame processing
                imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                            processFrame(imageProxy)
                        }
                    }

                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this,
                    cameraSelector,
                    imageAnalysis
                )

                Log.i(TAG, "Camera started (background mode)")

            } catch (e: Exception) {
                Log.e(TAG, "Camera start failed", e)
                updateNotification("Error: Camera failed")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        try {
            cameraProvider?.unbindAll()
            imageAnalysis = null
            Log.i(TAG, "Camera stopped")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping camera", e)
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (!isMonitoring) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmap()
            val timestamp = imageProxy.imageInfo.timestamp

            // Store last frame for VLM analysis
            lastFrame = bitmap
            lastFrameTimestamp = System.currentTimeMillis()

            // Update FPS
            val now = System.currentTimeMillis()
            frameCountForFps++
            if (now - fpsUpdateTime >= 1000) {
                val fps = frameCountForFps * 1000f / (now - fpsUpdateTime)
                _monitoringState.update { it.copy(fps = fps) }
                frameCountForFps = 0
                fpsUpdateTime = now
            }

            // Run fast pipeline
            val result = fastPipeline.processFrame(bitmap)

            // Update state
            _monitoringState.update { state ->
                state.copy(
                    personDetected = result.personDetected,
                    currentPose = result.pose,
                    motionLevel = result.motionLevel,
                    secondsSinceMotion = result.secondsSinceLastMotion,
                    frameCount = fastPipeline.getFrameCount()
                )
            }

        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Internal frame processing - shared by processFrame and processExternalFrame
     */
    // CLIP classification runs every N frames to save CPU
    private var clipFrameCounter = 0
    private val CLIP_FRAME_INTERVAL = 5  // Run CLIP every 5 frames

    private fun processFrameInternal(bitmap: Bitmap) {
        try {
            // Update FPS
            val now = System.currentTimeMillis()
            frameCountForFps++
            if (now - fpsUpdateTime >= 1000) {
                val fps = frameCountForFps * 1000f / (now - fpsUpdateTime)
                _monitoringState.update { it.copy(fps = fps) }
                frameCountForFps = 0
                fpsUpdateTime = now
            }

            // Run fast pipeline (YOLO)
            val result = fastPipeline.processFrame(bitmap)

            // Update state with YOLO results
            _monitoringState.update { state ->
                state.copy(
                    personDetected = result.personDetected,
                    currentPose = result.pose,
                    poseConfidence = result.poseConfidence,
                    motionLevel = result.motionLevel,
                    secondsSinceMotion = result.secondsSinceLastMotion,
                    frameCount = fastPipeline.getFrameCount(),
                    landmarks = result.landmarks
                )
            }

            // Run CLIP classification every N frames (when person detected)
            clipFrameCounter++
            if (clipFrameCounter >= CLIP_FRAME_INTERVAL && result.personDetected && fastClassifier.isReady()) {
                clipFrameCounter = 0
                serviceScope.launch {
                    runClipClassification(bitmap)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}")
        }
    }

    /**
     * Run CLIP classification on current frame
     */
    private suspend fun runClipClassification(bitmap: Bitmap) {
        try {
            val clipResult = fastClassifier.classify(bitmap)

            _monitoringState.update { state ->
                state.copy(
                    clipPosition = clipResult.position,
                    clipPositionConfidence = clipResult.positionConfidence,
                    clipAlertness = clipResult.alertness,
                    clipAlertnessConfidence = clipResult.alertnessConfidence,
                    clipActivity = clipResult.activity,
                    clipActivityConfidence = clipResult.activityConfidence,
                    clipSafety = clipResult.safety,
                    clipSafetyConfidence = clipResult.safetyConfidence,
                    clipInferenceMs = clipResult.inferenceTimeMs
                )
            }

            // Log classification result periodically
            if (fastPipeline.getFrameCount() % 30 == 0L) {
                Log.d(TAG, "CLIP: ${clipResult.toSummary()} (${clipResult.inferenceTimeMs}ms)")
            }

            // Check for safety concerns
            clipResult.getPrimaryConcern()?.let { concern ->
                Log.w(TAG, "CLIP safety concern: $concern")
            }

        } catch (e: Exception) {
            Log.e(TAG, "CLIP classification error: ${e.message}")
        }
    }

    private fun startVlmScheduler() {
        vlmSchedulerJob?.cancel()
        vlmSchedulerJob = serviceScope.launch {
            Log.i(TAG, "VLM scheduler started - first analysis in ${VLM_INTERVAL_MS}ms")

            while (isMonitoring) {
                delay(VLM_INTERVAL_MS)

                if (isMonitoring && !_monitoringState.value.isAnalyzing) {
                    Log.i(TAG, "VLM scheduler triggering analysis, lastFrame=${lastFrame != null}")
                    runVlmAnalysis("interval")
                }
            }

            Log.i(TAG, "VLM scheduler stopped")
        }
    }

    private fun collectAlerts() {
        alertCollectorJob?.cancel()
        alertCollectorJob = serviceScope.launch {
            fastPipeline.alerts.filterNotNull().collect { alert ->
                handleAlert(alert)
            }
        }
    }

    private suspend fun handleAlert(alert: FastPipeline.Alert) {
        Log.w(TAG, "Alert received: $alert")

        // Emit to UI
        _alerts.emit(alert)

        // Save alert to database
        val alertEntity = when (alert) {
            is FastPipeline.Alert.FallDetected -> AlertEntity(
                alertType = "fall",
                details = "Fall detected"
            )
            is FastPipeline.Alert.DepthVerifiedFall -> AlertEntity(
                alertType = "depth_fall",
                details = "Fall verified by depth: drop=${alert.dropMeters}m, confidence=${(alert.confidence * 100).toInt()}%"
            )
            is FastPipeline.Alert.LeavingBedZone -> AlertEntity(
                alertType = "leaving_bed",
                details = "Patient leaving bed zone: ${alert.distanceMeters}m from bed"
            )
            is FastPipeline.Alert.Stillness -> AlertEntity(
                alertType = "stillness",
                details = "No movement for ${alert.durationSeconds} seconds"
            )
            is FastPipeline.Alert.PoseChange -> AlertEntity(
                alertType = "pose_change",
                details = "Position changed from ${alert.from} to ${alert.to}"
            )
        }

        withContext(Dispatchers.IO) {
            app.database.alertDao().insert(alertEntity)
        }

        // Send alert notification
        val (title, message) = when (alert) {
            is FastPipeline.Alert.FallDetected ->
                "FALL DETECTED" to "Immediate attention required"
            is FastPipeline.Alert.DepthVerifiedFall ->
                "FALL DETECTED (Verified)" to "Drop: %.1fm, Confidence: %.0f%%".format(
                    alert.dropMeters, alert.confidence * 100
                )
            is FastPipeline.Alert.LeavingBedZone ->
                "Patient Leaving Bed" to "Distance: %.1fm".format(alert.distanceMeters)
            is FastPipeline.Alert.Stillness ->
                "Stillness Alert" to "No movement for ${alert.durationSeconds / 60} minutes"
            is FastPipeline.Alert.PoseChange ->
                "Position Change" to "${alert.from} → ${alert.to}"
        }

        sendAlertNotification(title, message)

        // Trigger VLM for context
        if (!_monitoringState.value.isAnalyzing) {
            val triggeredBy = when (alert) {
                is FastPipeline.Alert.FallDetected -> "fall"
                is FastPipeline.Alert.DepthVerifiedFall -> "depth_fall"
                is FastPipeline.Alert.LeavingBedZone -> "leaving_bed"
                is FastPipeline.Alert.Stillness -> "stillness"
                is FastPipeline.Alert.PoseChange -> "pose_change"
            }
            runVlmAnalysis(triggeredBy)
        }

        // Clear alert after handling
        fastPipeline.clearAlert()
    }

    private suspend fun runVlmAnalysis(triggeredBy: String) {
        val frame = lastFrame
        if (frame == null) {
            Log.w(TAG, "VLM analysis skipped - no frame available (frames received: ${_monitoringState.value.frameCount})")
            return
        }

        Log.i(TAG, "VLM analysis starting (triggered by: $triggeredBy, frame: ${frame.width}x${frame.height})")
        _monitoringState.update { it.copy(isAnalyzing = true) }

        try {
            val startTime = System.currentTimeMillis()
            val observation = withContext(Dispatchers.Default) {
                slowPipeline.analyzeFrame(frame)
            }
            val elapsed = System.currentTimeMillis() - startTime

            Log.i(TAG, "VLM analysis complete in ${elapsed}ms:")
            Log.i(TAG, "  position: ${observation.position}")
            Log.i(TAG, "  alertness: ${observation.alertness}")
            Log.i(TAG, "  chartNote: ${observation.chartNote}")

            // Save to database
            val entity = app.observationRepository.saveObservation(
                observation = observation,
                patientId = currentPatientId,
                triggeredBy = triggeredBy,
                motionLevel = _monitoringState.value.motionLevel,
                secondsStill = _monitoringState.value.secondsSinceMotion
            )

            // Update state
            _monitoringState.update { it.copy(lastObservation = observation) }
            Log.i(TAG, "VLM observation state updated")

            // Update notification with latest observation
            updateNotification("${observation.position} - ${observation.alertness}")

        } catch (e: Exception) {
            Log.e(TAG, "VLM analysis failed: ${e.message}", e)
        } finally {
            _monitoringState.update { it.copy(isAnalyzing = false) }
        }
    }

    /**
     * Trigger VLM analysis manually
     */
    fun triggerVlmAnalysis() {
        serviceScope.launch {
            runVlmAnalysis("manual")
        }
    }

    /**
     * Set current patient ID
     */
    fun setPatientId(patientId: String?) {
        currentPatientId = patientId
        Log.i(TAG, "Patient ID set: $patientId")
    }

    private fun createMonitoringNotification(text: String): Notification {
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val openPendingIntent = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val stopIntent = Intent(this, MonitoringService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, TriageVisionApp.CHANNEL_MONITORING)
            .setContentTitle(getString(R.string.notification_monitoring_title))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .setContentIntent(openPendingIntent)
            .addAction(
                android.R.drawable.ic_media_pause,
                "Stop",
                stopPendingIntent
            )
            .build()
    }

    private fun updateNotification(text: String) {
        val notification = createMonitoringNotification(text)
        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    private fun sendAlertNotification(title: String, message: String) {
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val openPendingIntent = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification = NotificationCompat.Builder(this, TriageVisionApp.CHANNEL_ALERTS)
            .setContentTitle(title)
            .setContentText(message)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setAutoCancel(true)
            .setContentIntent(openPendingIntent)
            .build()

        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(ALERT_NOTIFICATION_ID, notification)
    }

    private fun acquireWakeLock() {
        if (wakeLock == null) {
            val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "TriageVision::MonitoringWakeLock"
            ).apply {
                acquire(WAKE_LOCK_TIMEOUT)
            }
            Log.i(TAG, "Wake lock acquired")
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
                Log.i(TAG, "Wake lock released")
            }
        }
        wakeLock = null
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        super.onDestroy()

        serviceScope.cancel()
        cameraExecutor.shutdown()
        releaseWakeLock()

        lifecycleRegistry.currentState = Lifecycle.State.DESTROYED

        Log.i(TAG, "MonitoringService destroyed")
    }
}
