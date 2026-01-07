package com.triage.vision.pipeline

import android.graphics.Bitmap
import com.triage.vision.native.NativeBridge
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Fast Pipeline: Continuous motion/pose detection
 *
 * Uses NCNN + YOLO11n for real-time analysis at 15-30 FPS.
 * Triggers alerts and VLM analysis when conditions are met.
 */
class FastPipeline(
    private val nativeBridge: NativeBridge,
    private val config: FastPipelineConfig = FastPipelineConfig()
) {

    @Serializable
    data class FastPipelineConfig(
        val stillnessAlertSeconds: Int = 1800, // 30 minutes
        val fallDetectionEnabled: Boolean = true,
        val poseTrackingEnabled: Boolean = true
    )

    @Serializable
    data class DetectionResult(
        val personDetected: Boolean = false,
        val pose: Pose = Pose.UNKNOWN,
        val motionLevel: Float = 0f,
        val fallDetected: Boolean = false,
        val secondsSinceLastMotion: Long = 0
    )

    enum class Pose {
        UNKNOWN, LYING, SITTING, STANDING, FALLEN
    }

    sealed class Alert {
        data class Stillness(val durationSeconds: Long) : Alert()
        object FallDetected : Alert()
        data class PoseChange(val from: Pose, val to: Pose) : Alert()
    }

    private val _detectionState = MutableStateFlow(DetectionResult())
    val detectionState: StateFlow<DetectionResult> = _detectionState

    private val _alerts = MutableStateFlow<Alert?>(null)
    val alerts: StateFlow<Alert?> = _alerts

    private var lastMotionTimestamp = System.currentTimeMillis()
    private var lastPose = Pose.UNKNOWN
    private var frameCount = 0L

    /**
     * Process a camera frame
     * @param bitmap Camera frame (expects 640x480 or similar)
     * @return Detection result
     */
    fun processFrame(bitmap: Bitmap): DetectionResult {
        frameCount++

        // Run native detection
        val resultJson = nativeBridge.detectMotion(bitmap)
        val personDetected = nativeBridge.isPersonDetected(bitmap)
        val motionLevel = nativeBridge.getMotionLevel()

        // Update motion timestamp
        if (motionLevel > 0.1f) {
            lastMotionTimestamp = System.currentTimeMillis()
        }

        val secondsSinceMotion = (System.currentTimeMillis() - lastMotionTimestamp) / 1000

        // Parse pose from detection result
        val currentPose = parsePose(resultJson)
        val fallDetected = currentPose == Pose.FALLEN

        // Create result
        val result = DetectionResult(
            personDetected = personDetected,
            pose = currentPose,
            motionLevel = motionLevel,
            fallDetected = fallDetected,
            secondsSinceLastMotion = secondsSinceMotion
        )

        // Check for alerts
        checkAlerts(result)

        // Update state
        _detectionState.value = result
        lastPose = currentPose

        return result
    }

    private fun parsePose(json: String?): Pose {
        // TODO: Parse actual pose from YOLO detection classes
        return Pose.UNKNOWN
    }

    private fun checkAlerts(result: DetectionResult) {
        // Fall detection
        if (config.fallDetectionEnabled && result.fallDetected) {
            _alerts.value = Alert.FallDetected
            return
        }

        // Stillness alert
        if (result.secondsSinceLastMotion >= config.stillnessAlertSeconds) {
            _alerts.value = Alert.Stillness(result.secondsSinceLastMotion)
            return
        }

        // Pose change
        if (config.poseTrackingEnabled && result.pose != lastPose && lastPose != Pose.UNKNOWN) {
            _alerts.value = Alert.PoseChange(lastPose, result.pose)
        }
    }

    /**
     * Check if VLM analysis should be triggered
     */
    fun shouldTriggerVLM(): Boolean {
        val alert = _alerts.value
        return alert != null || _detectionState.value.secondsSinceLastMotion > 300 // 5 min stillness
    }

    /**
     * Clear current alert after handling
     */
    fun clearAlert() {
        _alerts.value = null
    }

    fun getFrameCount(): Long = frameCount
}
