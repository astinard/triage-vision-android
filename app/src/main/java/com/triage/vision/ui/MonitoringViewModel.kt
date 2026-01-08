package com.triage.vision.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
import android.os.IBinder
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.triage.vision.TriageVisionApp
import com.triage.vision.camera.DepthCameraManager
import com.triage.vision.data.AlertEntity
import com.triage.vision.data.ObservationEntity
import com.triage.vision.pipeline.FastPipeline
import com.triage.vision.pipeline.SlowPipeline
import com.triage.vision.service.MonitoringService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MonitoringViewModel : ViewModel() {

    companion object {
        private const val TAG = "MonitoringViewModel"
        private const val VLM_INTERVAL_MS = 2 * 1000L // 2 seconds
    }

    private val app = TriageVisionApp.instance
    private val fastPipeline = app.fastPipeline
    private val slowPipeline = app.slowPipeline
    private val chartEngine = app.chartEngine
    private val repository = app.observationRepository

    // Service connection
    private var monitoringService: MonitoringService? = null
    private var isBound = false
    private var serviceStateJob: Job? = null
    private var serviceAlertJob: Job? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            Log.i(TAG, "Service connected")
            val serviceBinder = binder as MonitoringService.MonitoringBinder
            monitoringService = serviceBinder.getService()
            isBound = true

            // Tell service that UI will feed frames
            monitoringService?.onUiBound()

            // Collect service state
            collectServiceState()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            Log.i(TAG, "Service disconnected")
            monitoringService?.onUiUnbound()
            monitoringService = null
            isBound = false
            serviceStateJob?.cancel()
            serviceAlertJob?.cancel()
        }
    }

    // UI State
    data class UiState(
        val isMonitoring: Boolean = false,
        val isAnalyzing: Boolean = false,
        val personDetected: Boolean = false,
        val currentPose: FastPipeline.Pose = FastPipeline.Pose.UNKNOWN,
        val poseConfidence: Float = 0f,
        val motionLevel: Float = 0f,
        val secondsSinceMotion: Long = 0,
        val lastObservation: SlowPipeline.Observation? = null,
        val currentAlert: FastPipeline.Alert? = null,
        val frameCount: Long = 0,
        val fps: Float = 0f,
        val error: String? = null,
        // Pose landmarks for skeleton drawing (33 points)
        val landmarks: List<FastPipeline.PoseLandmark> = emptyList(),
        // Depth sensor fields
        val depthAvailable: Boolean = false,
        val depthEnabled: Boolean = false,
        val distanceMeters: Float = 0f,
        val bedProximityMeters: Float = 0f,
        val inBedZone: Boolean = false,
        // Service connection
        val isServiceBound: Boolean = false,
        val useBackgroundService: Boolean = true
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    // Observations from database
    val recentObservations: Flow<List<ObservationEntity>> =
        repository.getRecentObservations(50)

    // Frame processing (for foreground-only mode)
    private var frameProcessingJob: Job? = null
    private var vlmSchedulerJob: Job? = null
    private var lastFrameTime = 0L
    private var frameCountForFps = 0
    private var fpsUpdateTime = 0L
    private var lastFrame: Bitmap? = null

    // Current patient (set via barcode scan)
    private var currentPatientId: String? = null

    /**
     * Bind to the MonitoringService
     */
    fun bindToService(context: Context) {
        if (isBound) return

        val intent = Intent(context, MonitoringService::class.java)
        context.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        _uiState.update { it.copy(isServiceBound = true) }
    }

    /**
     * Unbind from the MonitoringService
     */
    fun unbindFromService(context: Context) {
        if (!isBound) return

        // Tell service that UI is disconnecting
        monitoringService?.onUiUnbound()

        try {
            context.unbindService(serviceConnection)
        } catch (e: Exception) {
            Log.w(TAG, "Error unbinding service: ${e.message}")
        }

        isBound = false
        monitoringService = null
        serviceStateJob?.cancel()
        serviceAlertJob?.cancel()
        _uiState.update { it.copy(isServiceBound = false) }
    }

    private fun collectServiceState() {
        val service = monitoringService ?: return

        // Collect monitoring state
        serviceStateJob = viewModelScope.launch {
            service.monitoringState.collect { serviceState ->
                _uiState.update { state ->
                    state.copy(
                        isMonitoring = serviceState.isMonitoring,
                        isAnalyzing = serviceState.isAnalyzing,
                        personDetected = serviceState.personDetected,
                        currentPose = serviceState.currentPose,
                        poseConfidence = serviceState.poseConfidence,
                        motionLevel = serviceState.motionLevel,
                        secondsSinceMotion = serviceState.secondsSinceMotion,
                        lastObservation = serviceState.lastObservation,
                        frameCount = serviceState.frameCount,
                        fps = serviceState.fps,
                        landmarks = serviceState.landmarks
                    )
                }
            }
        }

        // Collect alerts
        serviceAlertJob = viewModelScope.launch {
            service.alerts.collect { alert ->
                _uiState.update { it.copy(currentAlert = alert) }
            }
        }
    }

    /**
     * Start monitoring via service (background-capable)
     */
    fun startMonitoringService(context: Context) {
        Log.i(TAG, "Starting monitoring service")

        val intent = Intent(context, MonitoringService::class.java).apply {
            action = MonitoringService.ACTION_START
            currentPatientId?.let { putExtra("patient_id", it) }
        }
        context.startForegroundService(intent)

        // Bind to receive state updates
        bindToService(context)
    }

    /**
     * Stop monitoring service
     */
    fun stopMonitoringService(context: Context) {
        Log.i(TAG, "Stopping monitoring service")

        val intent = Intent(context, MonitoringService::class.java).apply {
            action = MonitoringService.ACTION_STOP
        }
        context.startService(intent)

        _uiState.update { it.copy(isMonitoring = false) }
    }

    /**
     * Start monitoring (uses service if useBackgroundService is true)
     */
    fun startMonitoring(context: Context? = null) {
        if (_uiState.value.useBackgroundService && context != null) {
            startMonitoringService(context)
        } else {
            startForegroundMonitoring()
        }
    }

    /**
     * Stop monitoring
     */
    fun stopMonitoring(context: Context? = null) {
        if (_uiState.value.useBackgroundService && context != null) {
            stopMonitoringService(context)
        } else {
            stopForegroundMonitoring()
        }
    }

    /**
     * Start foreground-only monitoring (no service)
     */
    private fun startForegroundMonitoring() {
        Log.i(TAG, "Starting foreground monitoring")
        _uiState.update { it.copy(isMonitoring = true, error = null) }

        // Start VLM scheduler
        startVlmScheduler()

        // Observe fast pipeline alerts
        viewModelScope.launch {
            fastPipeline.alerts.collect { alert ->
                alert?.let {
                    handleAlert(it)
                }
            }
        }
    }

    /**
     * Stop foreground-only monitoring
     */
    private fun stopForegroundMonitoring() {
        Log.i(TAG, "Stopping foreground monitoring")
        _uiState.update { it.copy(isMonitoring = false) }

        frameProcessingJob?.cancel()
        vlmSchedulerJob?.cancel()
    }

    /**
     * Toggle background service mode
     */
    fun setBackgroundServiceEnabled(enabled: Boolean) {
        _uiState.update { it.copy(useBackgroundService = enabled) }
    }

    /**
     * Process a camera frame through the fast pipeline (foreground mode)
     * When using background service, sends frame to service for processing
     */
    fun processFrame(bitmap: Bitmap) {
        // Always send frames to service when bound (even if isMonitoring hasn't propagated yet)
        if (isBound && monitoringService != null) {
            monitoringService?.processExternalFrame(bitmap)
            // If service is monitoring, we're done - service handles everything
            if (_uiState.value.isMonitoring) return
        }

        if (!_uiState.value.isMonitoring) return

        // Store frame for VLM
        lastFrame = bitmap

        viewModelScope.launch(Dispatchers.Default) {
            try {
                // Update FPS
                val now = System.currentTimeMillis()
                frameCountForFps++
                if (now - fpsUpdateTime >= 1000) {
                    val fps = frameCountForFps * 1000f / (now - fpsUpdateTime)
                    _uiState.update { it.copy(fps = fps) }
                    frameCountForFps = 0
                    fpsUpdateTime = now
                }

                // Run fast pipeline
                val result = fastPipeline.processFrame(bitmap)

                // Update UI state with landmarks
                _uiState.update { state ->
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

                lastFrameTime = now

            } catch (e: Exception) {
                Log.e(TAG, "Frame processing error: ${e.message}", e)
            }
        }
    }

    /**
     * Process a camera frame with depth data through the fast pipeline
     */
    fun processFrameWithDepth(bitmap: Bitmap, depthFrame: DepthCameraManager.DepthFrame) {
        if (!_uiState.value.isMonitoring) return

        // If using service, let service handle it
        if (_uiState.value.useBackgroundService && isBound) {
            return
        }

        // Store frame for VLM
        lastFrame = bitmap

        viewModelScope.launch(Dispatchers.Default) {
            try {
                // Update FPS
                val now = System.currentTimeMillis()
                frameCountForFps++
                if (now - fpsUpdateTime >= 1000) {
                    val fps = frameCountForFps * 1000f / (now - fpsUpdateTime)
                    _uiState.update { it.copy(fps = fps) }
                    frameCountForFps = 0
                    fpsUpdateTime = now
                }

                // Run fast pipeline with depth
                val result = fastPipeline.processFrameWithDepth(bitmap, depthFrame)

                // Update UI state with depth metrics
                _uiState.update { state ->
                    state.copy(
                        personDetected = result.personDetected,
                        currentPose = result.pose,
                        motionLevel = result.motionLevel,
                        secondsSinceMotion = result.secondsSinceLastMotion,
                        frameCount = fastPipeline.getFrameCount(),
                        depthAvailable = result.depthAvailable,
                        distanceMeters = result.distanceMeters,
                        bedProximityMeters = result.bedProximityMeters,
                        inBedZone = result.inBedZone
                    )
                }

                lastFrameTime = now

            } catch (e: Exception) {
                Log.e(TAG, "Frame processing with depth error: ${e.message}", e)
            }
        }
    }

    /**
     * Set depth sensor availability
     */
    fun setDepthEnabled(enabled: Boolean) {
        _uiState.update { it.copy(depthEnabled = enabled) }
        Log.i(TAG, "Depth sensor enabled: $enabled")
    }

    /**
     * Trigger VLM analysis manually or on schedule
     */
    fun triggerVlmAnalysis(triggeredBy: String = "manual") {
        // If using service, delegate to service
        if (_uiState.value.useBackgroundService && isBound) {
            monitoringService?.triggerVlmAnalysis()
            return
        }

        if (_uiState.value.isAnalyzing) {
            Log.w(TAG, "VLM analysis already in progress")
            return
        }

        viewModelScope.launch {
            runVlmAnalysis(triggeredBy)
        }
    }

    private suspend fun runVlmAnalysis(triggeredBy: String) {
        Log.i(TAG, "Running VLM analysis (triggered by: $triggeredBy)")
        _uiState.update { it.copy(isAnalyzing = true) }

        try {
            val bitmap = lastFrame

            if (bitmap != null) {
                val observation = withContext(Dispatchers.Default) {
                    slowPipeline.analyzeFrame(bitmap)
                }

                Log.i(TAG, "VLM analysis complete: ${observation.chartNote}")

                // Save to database
                val entity = repository.saveObservation(
                    observation = observation,
                    patientId = currentPatientId,
                    triggeredBy = triggeredBy,
                    motionLevel = _uiState.value.motionLevel,
                    secondsStill = _uiState.value.secondsSinceMotion
                )

                // Update UI
                _uiState.update { it.copy(lastObservation = observation) }

                // Generate readable note for logging
                val chartEntry = chartEngine.generateChartEntry(observation)
                Log.i(TAG, "Chart note:\n${chartEngine.generateReadableNote(chartEntry)}")

            } else {
                Log.w(TAG, "No frame available for VLM analysis")
            }

        } catch (e: Exception) {
            Log.e(TAG, "VLM analysis failed: ${e.message}", e)
            _uiState.update { it.copy(error = "Analysis failed: ${e.message}") }
        } finally {
            _uiState.update { it.copy(isAnalyzing = false) }
        }
    }

    private fun startVlmScheduler() {
        vlmSchedulerJob?.cancel()
        vlmSchedulerJob = viewModelScope.launch {
            while (_uiState.value.isMonitoring) {
                delay(VLM_INTERVAL_MS)

                if (_uiState.value.isMonitoring && !_uiState.value.isAnalyzing) {
                    runVlmAnalysis("interval")
                }
            }
        }
    }

    private fun handleAlert(alert: FastPipeline.Alert) {
        Log.w(TAG, "Alert received: $alert")
        _uiState.update { it.copy(currentAlert = alert) }

        viewModelScope.launch {
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

            app.database.alertDao().insert(alertEntity)

            // Trigger VLM analysis for context
            if (!_uiState.value.isAnalyzing) {
                val triggeredBy = when (alert) {
                    is FastPipeline.Alert.FallDetected -> "fall"
                    is FastPipeline.Alert.DepthVerifiedFall -> "depth_fall"
                    is FastPipeline.Alert.LeavingBedZone -> "leaving_bed"
                    is FastPipeline.Alert.Stillness -> "stillness"
                    is FastPipeline.Alert.PoseChange -> "pose_change"
                }
                runVlmAnalysis(triggeredBy)
            }
        }

        // Clear alert after handling
        fastPipeline.clearAlert()
    }

    /**
     * Set current patient ID (from barcode scan)
     */
    fun setPatientId(patientId: String?) {
        currentPatientId = patientId
        monitoringService?.setPatientId(patientId)
        Log.i(TAG, "Patient ID set: $patientId")
    }

    /**
     * Acknowledge current alert
     */
    fun acknowledgeAlert() {
        _uiState.update { it.copy(currentAlert = null) }
    }

    /**
     * Clear error message
     */
    fun clearError() {
        _uiState.update { it.copy(error = null) }
    }

    /**
     * Export observations to FHIR JSON
     */
    suspend fun exportObservations(): String {
        val observations = repository.getUnexportedObservations()
        val fhirResources = observations.mapNotNull { it.fhirJson }

        // Mark as exported
        observations.forEach { repository.markExported(it.id) }

        return """{"resourceType": "Bundle", "type": "collection", "entry": [${fhirResources.joinToString(",")}]}"""
    }

    override fun onCleared() {
        super.onCleared()
        stopForegroundMonitoring()
        serviceStateJob?.cancel()
        serviceAlertJob?.cancel()
    }
}
