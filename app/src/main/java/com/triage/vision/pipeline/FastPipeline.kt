package com.triage.vision.pipeline

import android.graphics.Bitmap
import com.triage.vision.camera.DepthCameraManager
import com.triage.vision.camera.MediaPipePoseDetector
import com.triage.vision.native.NativeBridge
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Fast Pipeline: Continuous motion/pose detection
 *
 * Uses NCNN + YOLO11n for person detection and motion analysis.
 * Uses MediaPipe for accurate 33-point pose estimation.
 * Triggers alerts and VLM analysis when conditions are met.
 */
class FastPipeline(
    private val nativeBridge: NativeBridge,
    private val poseDetector: MediaPipePoseDetector? = null,
    private val config: FastPipelineConfig = FastPipelineConfig()
) {

    @Serializable
    data class FastPipelineConfig(
        val stillnessAlertSeconds: Int = 30, // 30 seconds
        val fallDetectionEnabled: Boolean = true,
        val poseTrackingEnabled: Boolean = true
    )

    @Serializable
    data class DetectionResult(
        val personDetected: Boolean = false,
        val pose: Pose = Pose.UNKNOWN,
        val poseConfidence: Float = 0f,
        val motionLevel: Float = 0f,
        val fallDetected: Boolean = false,
        val secondsSinceLastMotion: Long = 0,
        // Pose landmarks for skeleton drawing (33 points)
        val landmarks: List<PoseLandmark> = emptyList(),
        // Depth-enhanced fields
        val depthAvailable: Boolean = false,
        val depthFallDetected: Boolean = false,
        val verticalDropMeters: Float = 0f,
        val fallConfidence: Float = 0f,
        val distanceMeters: Float = 0f,
        val depthMotionLevel: Float = 0f,
        val bedProximityMeters: Float = 0f,
        val inBedZone: Boolean = false,
        val position3D: Position3D? = null
    )

    @Serializable
    data class PoseLandmark(
        val x: Float,  // 0-1 normalized
        val y: Float,  // 0-1 normalized
        val visibility: Float
    )

    @Serializable
    data class Position3D(
        val x: Float = 0f,
        val y: Float = 0f,
        val z: Float = 0f
    )

    enum class Pose {
        UNKNOWN, LYING, SITTING, STANDING, FALLEN
    }

    sealed class Alert {
        data class Stillness(val durationSeconds: Long) : Alert()
        object FallDetected : Alert()
        data class PoseChange(val from: Pose, val to: Pose) : Alert()
        // Depth-enhanced alerts
        data class DepthVerifiedFall(
            val dropMeters: Float,
            val confidence: Float
        ) : Alert()
        data class LeavingBedZone(val distanceMeters: Float) : Alert()
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

        // Run native detection for person detection and motion
        val resultJson = nativeBridge.detectMotion(bitmap)
        val personDetected = nativeBridge.isPersonDetected(bitmap)
        val motionLevel = nativeBridge.getMotionLevel()

        // Update motion timestamp
        if (motionLevel > 0.1f) {
            lastMotionTimestamp = System.currentTimeMillis()
        }

        val secondsSinceMotion = (System.currentTimeMillis() - lastMotionTimestamp) / 1000

        // Run MediaPipe pose detection for accurate pose and skeleton
        // Note: MediaPipe crashes on older devices (Pixel 2), so we catch errors gracefully
        var currentPose = Pose.UNKNOWN
        var poseConfidence = 0f
        var landmarks = emptyList<PoseLandmark>()

        if (personDetected && poseDetector != null) {
            try {
                val poseResult = poseDetector.detect(bitmap)
                if (poseResult.detected) {
                    currentPose = when (poseResult.pose) {
                        MediaPipePoseDetector.Pose.LYING -> Pose.LYING
                        MediaPipePoseDetector.Pose.SITTING -> Pose.SITTING
                        MediaPipePoseDetector.Pose.STANDING -> Pose.STANDING
                        MediaPipePoseDetector.Pose.FALLEN -> Pose.FALLEN
                        else -> Pose.UNKNOWN
                    }
                    poseConfidence = poseResult.confidence
                    landmarks = poseResult.landmarks.map { lm ->
                        PoseLandmark(lm.x, lm.y, lm.visibility)
                    }
                }
            } catch (e: Exception) {
                // MediaPipe may crash on older devices - continue without pose
                android.util.Log.w("FastPipeline", "Pose detection failed: ${e.message}")
            }
        }

        val fallDetected = currentPose == Pose.FALLEN

        // Create result
        val result = DetectionResult(
            personDetected = personDetected,
            pose = currentPose,
            poseConfidence = poseConfidence,
            motionLevel = motionLevel,
            fallDetected = fallDetected,
            secondsSinceLastMotion = secondsSinceMotion,
            landmarks = landmarks
        )

        // Check for alerts
        checkAlerts(result)

        // Update state
        _detectionState.value = result
        lastPose = currentPose

        return result
    }

    /**
     * Process a camera frame with depth data
     * @param bitmap RGB camera frame
     * @param depthFrame Depth frame from ToF sensor
     * @return Detection result with depth metrics
     */
    fun processFrameWithDepth(
        bitmap: Bitmap,
        depthFrame: DepthCameraManager.DepthFrame
    ): DetectionResult {
        frameCount++

        // Run native detection with depth
        val resultJson = nativeBridge.detectMotionWithDepth(
            bitmap,
            depthFrame.depthData,
            depthFrame.width,
            depthFrame.height
        )

        // Update motion timestamp
        val motionLevel = nativeBridge.getMotionLevel()
        if (motionLevel > 0.1f) {
            lastMotionTimestamp = System.currentTimeMillis()
        }

        val secondsSinceMotion = (System.currentTimeMillis() - lastMotionTimestamp) / 1000

        // Parse full result including depth metrics
        val result = parseDepthResult(resultJson, secondsSinceMotion)

        // Check for depth-enhanced alerts
        checkDepthAlerts(result)

        // Update state
        _detectionState.value = result
        lastPose = result.pose

        return result
    }

    private fun parseDepthResult(json: String?, secondsSinceMotion: Long): DetectionResult {
        if (json.isNullOrEmpty()) {
            return DetectionResult(secondsSinceLastMotion = secondsSinceMotion)
        }

        return try {
            // Parse JSON manually for now (could use kotlinx.serialization)
            val personDetected = json.contains("\"person_detected\": true")
            val fallDetected = json.contains("\"fall_detected\": true")
            val depthFall = json.contains("\"depth_fall\": true")
            val depthAvailable = json.contains("\"depth_available\": true")
            val inBedZone = json.contains("\"in_bed_zone\": true")

            // Extract numeric values
            val distanceMeters = extractFloat(json, "distance_meters") ?: 0f
            val verticalDrop = extractFloat(json, "vertical_drop_meters") ?: 0f
            val fallConfidence = extractFloat(json, "fall_confidence") ?: 0f
            val depthMotionLevel = extractFloat(json, "depth_motion_level") ?: 0f
            val bedProximity = extractFloat(json, "bed_proximity_meters") ?: 0f
            val motionLevel = extractFloat(json, "motion_level") ?: 0f

            // Extract 3D position
            val posX = extractFloat(json, "\"x\":") ?: 0f
            val posY = extractFloat(json, "\"y\":") ?: 0f
            val posZ = extractFloat(json, "\"z\":") ?: 0f

            val pose = parsePose(json)

            DetectionResult(
                personDetected = personDetected,
                pose = pose,
                motionLevel = motionLevel,
                fallDetected = fallDetected,
                secondsSinceLastMotion = secondsSinceMotion,
                depthAvailable = depthAvailable,
                depthFallDetected = depthFall,
                verticalDropMeters = verticalDrop,
                fallConfidence = fallConfidence,
                distanceMeters = distanceMeters,
                depthMotionLevel = depthMotionLevel,
                bedProximityMeters = bedProximity,
                inBedZone = inBedZone,
                position3D = Position3D(posX, posY, posZ)
            )
        } catch (e: Exception) {
            DetectionResult(secondsSinceLastMotion = secondsSinceMotion)
        }
    }

    private fun extractFloat(json: String, key: String): Float? {
        val pattern = "$key\\s*[\":]?\\s*(-?\\d+\\.?\\d*)".toRegex()
        return pattern.find(json)?.groupValues?.get(1)?.toFloatOrNull()
    }

    private fun checkDepthAlerts(result: DetectionResult) {
        // Depth-verified fall (highest priority)
        if (result.depthFallDetected && result.fallConfidence > 0.7f) {
            _alerts.value = Alert.DepthVerifiedFall(
                dropMeters = result.verticalDropMeters,
                confidence = result.fallConfidence
            )
            return
        }

        // Regular fall detection
        if (config.fallDetectionEnabled && result.fallDetected) {
            _alerts.value = Alert.FallDetected
            return
        }

        // Leaving bed zone alert
        if (result.depthAvailable && !result.inBedZone && result.bedProximityMeters > 0.5f) {
            // Only alert if they were previously in bed zone
            val prevResult = _detectionState.value
            if (prevResult.inBedZone) {
                _alerts.value = Alert.LeavingBedZone(result.bedProximityMeters)
                return
            }
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
