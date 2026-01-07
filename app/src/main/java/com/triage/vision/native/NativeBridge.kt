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
