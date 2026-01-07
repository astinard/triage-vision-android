package com.triage.vision.native

import android.graphics.Bitmap

/**
 * JNI bridge to native inference libraries (NCNN, llama.cpp)
 */
class NativeBridge {

    companion object {
        init {
            System.loadLibrary("triage_vision")
        }
    }

    /**
     * Initialize native libraries with model path
     * @param modelPath Path to model files directory
     * @return 0 on success, error code otherwise
     */
    external fun init(modelPath: String): Int

    /**
     * Fast Pipeline: Detect motion and pose in frame
     * @param bitmap Camera frame
     * @return Detection results (JSON string)
     */
    external fun detectMotion(bitmap: Bitmap): String?

    /**
     * Fast Pipeline: Quick check if person is in frame
     * @param bitmap Camera frame
     * @return true if person detected
     */
    external fun isPersonDetected(bitmap: Bitmap): Boolean

    /**
     * Fast Pipeline: Get current motion level
     * @return Motion level 0.0 (still) to 1.0 (active)
     */
    external fun getMotionLevel(): Float

    /**
     * Fast Pipeline: Detect motion and pose with depth enhancement
     * @param bitmap Camera frame (RGB)
     * @param depthData Depth map (DEPTH16 format, values in millimeters)
     * @param depthWidth Depth frame width
     * @param depthHeight Depth frame height
     * @return Detection results with depth metrics (JSON string)
     */
    external fun detectMotionWithDepth(
        bitmap: Bitmap,
        depthData: ShortArray,
        depthWidth: Int,
        depthHeight: Int
    ): String?

    /**
     * Get depth value at specific pixel coordinates
     * @param x X coordinate in depth frame
     * @param y Y coordinate in depth frame
     * @return Depth in meters, or -1 if invalid
     */
    external fun getDepthAt(x: Int, y: Int): Float

    /**
     * Get average distance to detected person
     * @return Distance in meters
     */
    external fun getAverageDistance(): Float

    /**
     * Slow Pipeline: Run VLM analysis on frame
     * @param bitmap Camera frame to analyze
     * @param prompt Analysis prompt
     * @return JSON observation result
     */
    external fun analyzeScene(bitmap: Bitmap, prompt: String): String

    /**
     * Cleanup native resources
     */
    external fun cleanup()
}
