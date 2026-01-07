package com.triage.vision.ui

import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.triage.vision.TriageVisionApp
import com.triage.vision.data.AlertEntity
import com.triage.vision.data.ObservationEntity
import com.triage.vision.pipeline.FastPipeline
import com.triage.vision.pipeline.SlowPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MonitoringViewModel : ViewModel() {

    companion object {
        private const val TAG = "MonitoringViewModel"
        private const val VLM_INTERVAL_MS = 15 * 60 * 1000L // 15 minutes
    }

    private val app = TriageVisionApp.instance
    private val fastPipeline = app.fastPipeline
    private val slowPipeline = app.slowPipeline
    private val chartEngine = app.chartEngine
    private val repository = app.observationRepository

    // UI State
    data class UiState(
        val isMonitoring: Boolean = false,
        val isAnalyzing: Boolean = false,
        val personDetected: Boolean = false,
        val currentPose: FastPipeline.Pose = FastPipeline.Pose.UNKNOWN,
        val motionLevel: Float = 0f,
        val secondsSinceMotion: Long = 0,
        val lastObservation: SlowPipeline.Observation? = null,
        val currentAlert: FastPipeline.Alert? = null,
        val frameCount: Long = 0,
        val fps: Float = 0f,
        val error: String? = null
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    // Observations from database
    val recentObservations: Flow<List<ObservationEntity>> =
        repository.getRecentObservations(50)

    // Frame processing
    private var frameProcessingJob: Job? = null
    private var vlmSchedulerJob: Job? = null
    private var lastFrameTime = 0L
    private var frameCountForFps = 0
    private var fpsUpdateTime = 0L

    // Current patient (set via barcode scan)
    private var currentPatientId: String? = null

    /**
     * Start monitoring
     */
    fun startMonitoring() {
        Log.i(TAG, "Starting monitoring")
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
     * Stop monitoring
     */
    fun stopMonitoring() {
        Log.i(TAG, "Stopping monitoring")
        _uiState.update { it.copy(isMonitoring = false) }

        frameProcessingJob?.cancel()
        vlmSchedulerJob?.cancel()
    }

    /**
     * Process a camera frame through the fast pipeline
     */
    fun processFrame(bitmap: Bitmap) {
        if (!_uiState.value.isMonitoring) return

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

                // Update UI state
                _uiState.update { state ->
                    state.copy(
                        personDetected = result.personDetected,
                        currentPose = result.pose,
                        motionLevel = result.motionLevel,
                        secondsSinceMotion = result.secondsSinceLastMotion,
                        frameCount = fastPipeline.getFrameCount()
                    )
                }

                lastFrameTime = now

            } catch (e: Exception) {
                Log.e(TAG, "Frame processing error: ${e.message}", e)
            }
        }
    }

    /**
     * Trigger VLM analysis manually or on schedule
     */
    fun triggerVlmAnalysis(triggeredBy: String = "manual") {
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
            // TODO: Get current frame from camera
            // For now, this is a placeholder - in real implementation,
            // we'd capture a frame from the camera preview
            val bitmap = getCurrentFrame()

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

    // Placeholder - in real implementation, this would capture from CameraX
    private fun getCurrentFrame(): Bitmap? {
        // TODO: Implement frame capture from CameraX preview
        return null
    }

    override fun onCleared() {
        super.onCleared()
        stopMonitoring()
    }
}
