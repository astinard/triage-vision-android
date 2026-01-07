package com.triage.vision.pipeline

import android.graphics.Bitmap
import com.triage.vision.native.NativeBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Slow Pipeline: VLM-based scene understanding
 *
 * Uses llama.cpp + SmolVLM-500M for detailed scene analysis.
 * Runs on interval or when triggered by Fast Pipeline alerts.
 */
class SlowPipeline(
    private val nativeBridge: NativeBridge,
    private val config: SlowPipelineConfig = SlowPipelineConfig()
) {

    @Serializable
    data class SlowPipelineConfig(
        val intervalMinutes: Int = 15,
        val triggerOnStillness: Boolean = true,
        val triggerOnFall: Boolean = true
    )

    @Serializable
    data class Observation(
        val timestamp: String = "",
        val position: String = "unknown",
        val alertness: String = "unknown",
        val movementLevel: String = "unknown",
        val equipmentVisible: List<String> = emptyList(),
        val concerns: List<String> = emptyList(),
        val comfortAssessment: String = "unknown",
        val chartNote: String = ""
    )

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    companion object {
        const val DEFAULT_PROMPT = """Analyze this patient monitoring image. Describe:
1. Patient position (lying_supine, lying_left_lateral, lying_right_lateral, sitting, standing)
2. Alertness level (awake, sleeping, drowsy, eyes_closed, unresponsive)
3. Movement level (none, minimal, moderate, active)
4. Any visible medical equipment (iv_line, pulse_oximeter, nasal_cannula, feeding_tube, catheter, monitor_leads)
5. Any concerns or notable observations
6. General patient comfort assessment (comfortable, restless, in_distress, pain_indicated)

Respond ONLY with valid JSON in this exact format:
{
  "position": "string",
  "alertness": "string",
  "movement_level": "string",
  "equipment_visible": ["string"],
  "concerns": ["string"],
  "comfort_assessment": "string",
  "chart_note": "Brief nursing observation summary"
}"""
    }

    private var lastAnalysisTimestamp = 0L

    /**
     * Run VLM analysis on a frame
     * @param bitmap Camera frame to analyze
     * @param customPrompt Optional custom prompt (uses DEFAULT_PROMPT if null)
     * @return Observation result
     */
    suspend fun analyzeFrame(
        bitmap: Bitmap,
        customPrompt: String? = null
    ): Observation = withContext(Dispatchers.Default) {

        val prompt = customPrompt ?: DEFAULT_PROMPT
        val startTime = System.currentTimeMillis()

        // Run native VLM inference
        val resultJson = nativeBridge.analyzeScene(bitmap, prompt)

        val inferenceTime = System.currentTimeMillis() - startTime
        android.util.Log.i("SlowPipeline", "VLM inference took ${inferenceTime}ms")

        // Parse result
        val observation = parseObservation(resultJson)
        lastAnalysisTimestamp = System.currentTimeMillis()

        observation.copy(
            timestamp = java.time.Instant.now().toString()
        )
    }

    private fun parseObservation(jsonString: String): Observation {
        return try {
            // Try to parse the nested observation structure
            val parsed = json.decodeFromString<ObservationWrapper>(jsonString)
            parsed.observation ?: Observation(chartNote = "Failed to parse observation")
        } catch (e: Exception) {
            try {
                // Try direct parsing
                json.decodeFromString<Observation>(jsonString)
            } catch (e2: Exception) {
                android.util.Log.e("SlowPipeline", "Failed to parse VLM output: $jsonString", e2)
                Observation(
                    concerns = listOf("Parse error: ${e2.message}"),
                    chartNote = "VLM output could not be parsed"
                )
            }
        }
    }

    @Serializable
    private data class ObservationWrapper(
        val observation: Observation? = null
    )

    /**
     * Check if scheduled analysis is due
     */
    fun isAnalysisDue(): Boolean {
        val elapsed = System.currentTimeMillis() - lastAnalysisTimestamp
        val intervalMs = config.intervalMinutes * 60 * 1000L
        return elapsed >= intervalMs
    }

    /**
     * Get time until next scheduled analysis
     */
    fun getTimeUntilNextAnalysis(): Long {
        val elapsed = System.currentTimeMillis() - lastAnalysisTimestamp
        val intervalMs = config.intervalMinutes * 60 * 1000L
        return maxOf(0, intervalMs - elapsed)
    }
}
