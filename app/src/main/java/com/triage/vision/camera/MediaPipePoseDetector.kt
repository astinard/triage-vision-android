package com.triage.vision.camera

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * MediaPipe Pose Landmarker wrapper for accurate pose detection.
 *
 * Detects 33 body keypoints and classifies pose (lying, sitting, standing, fallen).
 * Runs on GPU when available for real-time performance.
 */
class MediaPipePoseDetector(private val context: Context) {

    companion object {
        private const val TAG = "MediaPipePoseDetector"

        // MediaPipe pose model path (must include directory to have a slash)
        private const val MODEL_NAME = "models/pose_landmarker_lite.task"

        // Landmark indices for key body parts
        const val NOSE = 0
        const val LEFT_SHOULDER = 11
        const val RIGHT_SHOULDER = 12
        const val LEFT_HIP = 23
        const val RIGHT_HIP = 24
        const val LEFT_KNEE = 25
        const val RIGHT_KNEE = 26
        const val LEFT_ANKLE = 27
        const val RIGHT_ANKLE = 28

        // Pose classification thresholds
        private const val LYING_THRESHOLD = 0.3f  // Hip-shoulder angle threshold
        private const val STANDING_THRESHOLD = 0.7f  // Vertical alignment threshold
    }

    private var poseLandmarker: PoseLandmarker? = null
    private var isInitialized = false

    /**
     * Pose detection result with keypoints and classification
     */
    data class PoseResult(
        val detected: Boolean = false,
        val landmarks: List<Landmark> = emptyList(),
        val pose: Pose = Pose.UNKNOWN,
        val confidence: Float = 0f
    )

    /**
     * Single landmark point
     */
    data class Landmark(
        val x: Float,  // 0-1 normalized
        val y: Float,  // 0-1 normalized
        val z: Float,  // Depth relative to hip
        val visibility: Float,
        val index: Int
    )

    /**
     * Pose classification
     */
    enum class Pose {
        UNKNOWN,
        LYING,
        SITTING,
        STANDING,
        FALLEN
    }

    /**
     * Initialize the pose detector
     * Uses CPU delegate for maximum compatibility (GPU crashes on older devices like Pixel 2)
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        try {
            Log.i(TAG, "Initializing MediaPipe Pose detector with CPU delegate...")

            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_NAME)
                .setDelegate(Delegate.CPU)  // Use CPU for compatibility
                .build()

            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumPoses(1)  // Single person detection
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(context, options)
            isInitialized = true
            Log.i(TAG, "MediaPipe Pose detector initialized successfully (CPU)")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize pose detector: ${e.message}", e)
            return false
        }
    }

    /**
     * Detect pose from bitmap
     */
    fun detect(bitmap: Bitmap): PoseResult {
        if (!isInitialized) {
            if (!initialize()) {
                return PoseResult()
            }
        }

        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = poseLandmarker?.detect(mpImage)

            if (result == null || result.landmarks().isEmpty()) {
                return PoseResult()
            }

            // Convert to our format
            val landmarks = result.landmarks()[0].map { landmark ->
                Landmark(
                    x = landmark.x(),
                    y = landmark.y(),
                    z = landmark.z(),
                    visibility = landmark.visibility().orElse(0f),
                    index = result.landmarks()[0].indexOf(landmark)
                )
            }

            // Classify pose from landmarks
            val (pose, confidence) = classifyPose(landmarks)

            PoseResult(
                detected = true,
                landmarks = landmarks,
                pose = pose,
                confidence = confidence
            )

        } catch (e: Exception) {
            Log.e(TAG, "Pose detection error: ${e.message}")
            PoseResult()
        }
    }

    /**
     * Classify pose based on landmark positions
     */
    private fun classifyPose(landmarks: List<Landmark>): Pair<Pose, Float> {
        if (landmarks.size < 33) return Pose.UNKNOWN to 0f

        val leftShoulder = landmarks[LEFT_SHOULDER]
        val rightShoulder = landmarks[RIGHT_SHOULDER]
        val leftHip = landmarks[LEFT_HIP]
        val rightHip = landmarks[RIGHT_HIP]
        val leftAnkle = landmarks[LEFT_ANKLE]
        val rightAnkle = landmarks[RIGHT_ANKLE]

        // Check visibility of key landmarks
        val keyVisibility = minOf(
            leftShoulder.visibility,
            rightShoulder.visibility,
            leftHip.visibility,
            rightHip.visibility
        )

        if (keyVisibility < 0.3f) return Pose.UNKNOWN to 0f

        // Calculate body orientation
        val shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2
        val hipMidY = (leftHip.y + rightHip.y) / 2
        val ankleMidY = (leftAnkle.y + rightAnkle.y) / 2

        // Vertical span (shoulder to ankle)
        val verticalSpan = kotlin.math.abs(ankleMidY - shoulderMidY)

        // Horizontal span (shoulders)
        val shoulderWidth = kotlin.math.abs(rightShoulder.x - leftShoulder.x)
        val hipWidth = kotlin.math.abs(rightHip.x - leftHip.x)

        // Torso angle (shoulder-hip vector)
        val torsoAngle = kotlin.math.atan2(
            (hipMidY - shoulderMidY).toDouble(),
            ((leftHip.x + rightHip.x) / 2 - (leftShoulder.x + rightShoulder.x) / 2).toDouble()
        )
        val torsoAngleDeg = kotlin.math.abs(Math.toDegrees(torsoAngle))

        // Classification logic
        return when {
            // Lying: body mostly horizontal (shoulder-hip near horizontal)
            torsoAngleDeg < 45 || torsoAngleDeg > 135 -> {
                // Check if it's a fall (sudden horizontal position at bottom of frame)
                if (shoulderMidY > 0.7f && hipMidY > 0.7f) {
                    Pose.FALLEN to 0.8f
                } else {
                    Pose.LYING to 0.85f
                }
            }

            // Standing: vertical alignment, ankles below hips below shoulders
            verticalSpan > 0.5f && shoulderMidY < hipMidY && hipMidY < ankleMidY -> {
                Pose.STANDING to 0.9f
            }

            // Sitting: hips and shoulders at similar height, ankles at different position
            kotlin.math.abs(shoulderMidY - hipMidY) < 0.2f -> {
                Pose.SITTING to 0.8f
            }

            // Default to standing if mostly vertical
            torsoAngleDeg > 60 && torsoAngleDeg < 120 -> {
                Pose.STANDING to 0.6f
            }

            else -> Pose.UNKNOWN to 0.3f
        }
    }

    /**
     * Get skeleton connections for drawing
     */
    fun getSkeletonConnections(): List<Pair<Int, Int>> = listOf(
        // Face
        0 to 1, 1 to 2, 2 to 3, 3 to 7,
        0 to 4, 4 to 5, 5 to 6, 6 to 8,
        9 to 10,

        // Torso
        11 to 12,  // Shoulders
        11 to 23, 12 to 24,  // Shoulders to hips
        23 to 24,  // Hips

        // Left arm
        11 to 13, 13 to 15, 15 to 17, 15 to 19, 15 to 21, 17 to 19,

        // Right arm
        12 to 14, 14 to 16, 16 to 18, 16 to 20, 16 to 22, 18 to 20,

        // Left leg
        23 to 25, 25 to 27, 27 to 29, 27 to 31, 29 to 31,

        // Right leg
        24 to 26, 26 to 28, 28 to 30, 28 to 32, 30 to 32
    )

    /**
     * Cleanup resources
     */
    fun close() {
        poseLandmarker?.close()
        poseLandmarker = null
        isInitialized = false
        Log.i(TAG, "MediaPipe Pose detector closed")
    }
}
