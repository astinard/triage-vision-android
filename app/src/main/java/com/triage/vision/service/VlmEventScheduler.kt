package com.triage.vision.service

import android.util.Log
import com.triage.vision.classifier.FastClassifier
import com.triage.vision.classifier.NursingLabels

/**
 * Smart VLM scheduler that triggers analysis based on CLIP classification events.
 *
 * Instead of running VLM on a fixed timer, this scheduler detects meaningful
 * state changes in the CLIP classification results and triggers VLM only when
 * clinically relevant events occur.
 *
 * Trigger conditions:
 * - Position change (e.g., lying → sitting, sitting → standing)
 * - Safety concern detected (fall risk, leaving bed, fallen)
 * - Alertness change (awake → drowsy → unresponsive)
 * - Activity spike (still → restless, or vice versa)
 * - Comfort concern (comfortable → distressed)
 * - Periodic fallback (every N minutes if no events)
 */
class VlmEventScheduler(
    private val onTriggerVlm: (reason: String) -> Unit
) {
    companion object {
        private const val TAG = "VlmEventScheduler"

        // Minimum time between VLM runs (prevent spam and memory pressure)
        // Increased to 60s for stability on low-memory devices like Pixel 2
        private const val MIN_VLM_INTERVAL_MS = 60_000L  // 60 seconds

        // Maximum time without VLM (fallback periodic check)
        private const val MAX_VLM_INTERVAL_MS = 5 * 60 * 1000L  // 5 minutes

        // Confidence threshold for state changes to be considered significant
        private const val CONFIDENCE_THRESHOLD = 0.4f

        // Safety concerns always trigger VLM above this confidence
        private const val SAFETY_TRIGGER_THRESHOLD = 0.5f
    }

    // Previous state for change detection
    private var previousPosition: NursingLabels.Position? = null
    private var previousAlertness: NursingLabels.Alertness? = null
    private var previousActivity: NursingLabels.Activity? = null
    private var previousSafety: NursingLabels.SafetyConcern? = null
    private var previousComfort: NursingLabels.Comfort? = null

    // Timing
    private var lastVlmTriggerTime = 0L
    private var lastClassificationTime = 0L

    // State stability tracking (require N consistent readings before triggering)
    private var stablePositionCount = 0
    private var stableAlertnessCount = 0
    private var stableActivityCount = 0
    private var lastStablePosition: NursingLabels.Position? = null
    private var lastStableAlertness: NursingLabels.Alertness? = null
    private var lastStableActivity: NursingLabels.Activity? = null

    private val STABILITY_THRESHOLD = 3  // Require 3 consistent readings

    /**
     * Event types that can trigger VLM
     */
    sealed class TriggerEvent(val reason: String, val priority: Int) {
        data class SafetyConcern(val concern: NursingLabels.SafetyConcern, val confidence: Float)
            : TriggerEvent("safety_${concern.name.lowercase()}", priority = 1)

        data class PositionChange(val from: NursingLabels.Position, val to: NursingLabels.Position)
            : TriggerEvent("position_${from.name.lowercase()}_to_${to.name.lowercase()}", priority = 2)

        data class AlertnessChange(val from: NursingLabels.Alertness, val to: NursingLabels.Alertness)
            : TriggerEvent("alertness_${from.name.lowercase()}_to_${to.name.lowercase()}", priority = 2)

        data class ActivityChange(val from: NursingLabels.Activity, val to: NursingLabels.Activity)
            : TriggerEvent("activity_${from.name.lowercase()}_to_${to.name.lowercase()}", priority = 3)

        data class ComfortConcern(val level: NursingLabels.Comfort, val confidence: Float)
            : TriggerEvent("comfort_${level.name.lowercase()}", priority = 2)

        object PeriodicFallback : TriggerEvent("periodic_fallback", priority = 4)
    }

    /**
     * Process a CLIP classification result and decide whether to trigger VLM.
     *
     * @param result The CLIP classification result
     * @return The trigger event if VLM should run, null otherwise
     */
    fun processClassification(result: FastClassifier.ClassificationResult): TriggerEvent? {
        val now = System.currentTimeMillis()
        lastClassificationTime = now

        val events = mutableListOf<TriggerEvent>()

        // Check for safety concerns (highest priority)
        if (result.safety != NursingLabels.SafetyConcern.NONE &&
            result.safetyConfidence >= SAFETY_TRIGGER_THRESHOLD) {
            events.add(TriggerEvent.SafetyConcern(result.safety, result.safetyConfidence))
        }

        // Check for comfort concerns
        if ((result.comfort == NursingLabels.Comfort.DISTRESSED ||
             result.comfort == NursingLabels.Comfort.PAIN_INDICATED) &&
            result.comfortConfidence >= CONFIDENCE_THRESHOLD) {
            events.add(TriggerEvent.ComfortConcern(result.comfort, result.comfortConfidence))
        }

        // Track position stability
        if (result.position == lastStablePosition) {
            stablePositionCount++
        } else {
            lastStablePosition = result.position
            stablePositionCount = 1
        }

        // Check for stable position change
        if (stablePositionCount >= STABILITY_THRESHOLD &&
            previousPosition != null &&
            result.position != previousPosition &&
            result.positionConfidence >= CONFIDENCE_THRESHOLD) {

            val change = TriggerEvent.PositionChange(previousPosition!!, result.position)
            if (isSignificantPositionChange(previousPosition!!, result.position)) {
                events.add(change)
            }
        }

        // Track alertness stability
        if (result.alertness == lastStableAlertness) {
            stableAlertnessCount++
        } else {
            lastStableAlertness = result.alertness
            stableAlertnessCount = 1
        }

        // Check for stable alertness change
        if (stableAlertnessCount >= STABILITY_THRESHOLD &&
            previousAlertness != null &&
            result.alertness != previousAlertness &&
            result.alertnessConfidence >= CONFIDENCE_THRESHOLD) {

            val change = TriggerEvent.AlertnessChange(previousAlertness!!, result.alertness)
            if (isSignificantAlertnessChange(previousAlertness!!, result.alertness)) {
                events.add(change)
            }
        }

        // Track activity stability
        if (result.activity == lastStableActivity) {
            stableActivityCount++
        } else {
            lastStableActivity = result.activity
            stableActivityCount = 1
        }

        // Check for stable activity change
        if (stableActivityCount >= STABILITY_THRESHOLD &&
            previousActivity != null &&
            result.activity != previousActivity &&
            result.activityConfidence >= CONFIDENCE_THRESHOLD) {

            val change = TriggerEvent.ActivityChange(previousActivity!!, result.activity)
            if (isSignificantActivityChange(previousActivity!!, result.activity)) {
                events.add(change)
            }
        }

        // Update previous state (only when stable)
        if (stablePositionCount >= STABILITY_THRESHOLD) {
            previousPosition = result.position
        }
        if (stableAlertnessCount >= STABILITY_THRESHOLD) {
            previousAlertness = result.alertness
        }
        if (stableActivityCount >= STABILITY_THRESHOLD) {
            previousActivity = result.activity
        }
        previousSafety = result.safety
        previousComfort = result.comfort

        // Check periodic fallback
        if (events.isEmpty() && (now - lastVlmTriggerTime) >= MAX_VLM_INTERVAL_MS) {
            events.add(TriggerEvent.PeriodicFallback)
        }

        // Select highest priority event
        val selectedEvent = events.minByOrNull { it.priority }

        // Check if we can trigger (respect minimum interval)
        if (selectedEvent != null) {
            if ((now - lastVlmTriggerTime) >= MIN_VLM_INTERVAL_MS) {
                lastVlmTriggerTime = now
                Log.i(TAG, "Triggering VLM: ${selectedEvent.reason}")
                onTriggerVlm(selectedEvent.reason)
                return selectedEvent
            } else {
                Log.d(TAG, "VLM trigger suppressed (too soon): ${selectedEvent.reason}")
            }
        }

        return null
    }

    /**
     * Check if enough time has passed for periodic fallback
     */
    fun shouldRunPeriodicFallback(): Boolean {
        val now = System.currentTimeMillis()
        return (now - lastVlmTriggerTime) >= MAX_VLM_INTERVAL_MS
    }

    /**
     * Force trigger VLM (for manual triggers or alerts)
     */
    fun forceTrigger(reason: String) {
        lastVlmTriggerTime = System.currentTimeMillis()
        onTriggerVlm(reason)
    }

    /**
     * Reset scheduler state (e.g., when monitoring starts)
     */
    fun reset() {
        previousPosition = null
        previousAlertness = null
        previousActivity = null
        previousSafety = null
        previousComfort = null
        lastVlmTriggerTime = 0L
        lastClassificationTime = 0L
        stablePositionCount = 0
        stableAlertnessCount = 0
        stableActivityCount = 0
        lastStablePosition = null
        lastStableAlertness = null
        lastStableActivity = null
        Log.i(TAG, "Scheduler reset")
    }

    /**
     * Determine if a position change is significant enough to trigger VLM.
     *
     * Significant changes:
     * - Any change involving ON_FLOOR (fall detection)
     * - Lying to sitting/standing (patient getting up)
     * - Standing/sitting to lying (patient going to bed)
     */
    private fun isSignificantPositionChange(
        from: NursingLabels.Position,
        to: NursingLabels.Position
    ): Boolean {
        // Any involvement of floor is significant
        if (from == NursingLabels.Position.ON_FLOOR || to == NursingLabels.Position.ON_FLOOR) {
            return true
        }

        // Lying to upright positions
        val lyingPositions = setOf(
            NursingLabels.Position.LYING_SUPINE,
            NursingLabels.Position.LYING_LEFT,
            NursingLabels.Position.LYING_RIGHT,
            NursingLabels.Position.LYING_PRONE
        )
        val uprightPositions = setOf(
            NursingLabels.Position.SITTING_BED,
            NursingLabels.Position.SITTING_CHAIR,
            NursingLabels.Position.STANDING
        )

        // Transition between lying and upright is significant
        if ((from in lyingPositions && to in uprightPositions) ||
            (from in uprightPositions && to in lyingPositions)) {
            return true
        }

        // Standing is always significant
        if (from == NursingLabels.Position.STANDING || to == NursingLabels.Position.STANDING) {
            return true
        }

        // Minor changes between lying positions are less significant
        return false
    }

    /**
     * Determine if an alertness change is significant.
     *
     * Significant changes:
     * - Any change involving UNRESPONSIVE
     * - AWAKE to SLEEPING or vice versa
     * - Multiple level jumps (e.g., AWAKE_ALERT to SLEEPING)
     */
    private fun isSignificantAlertnessChange(
        from: NursingLabels.Alertness,
        to: NursingLabels.Alertness
    ): Boolean {
        // Unresponsive is always significant
        if (from == NursingLabels.Alertness.UNRESPONSIVE ||
            to == NursingLabels.Alertness.UNRESPONSIVE) {
            return true
        }

        // Awake to sleeping or vice versa
        val awakeStates = setOf(
            NursingLabels.Alertness.AWAKE_ALERT,
            NursingLabels.Alertness.AWAKE_DROWSY
        )
        val sleepStates = setOf(
            NursingLabels.Alertness.SLEEPING,
            NursingLabels.Alertness.EYES_CLOSED
        )

        if ((from in awakeStates && to in sleepStates) ||
            (from in sleepStates && to in awakeStates)) {
            return true
        }

        // ALERT to DROWSY or vice versa
        if ((from == NursingLabels.Alertness.AWAKE_ALERT && to == NursingLabels.Alertness.AWAKE_DROWSY) ||
            (from == NursingLabels.Alertness.AWAKE_DROWSY && to == NursingLabels.Alertness.AWAKE_ALERT)) {
            return true
        }

        return false
    }

    /**
     * Determine if an activity change is significant.
     *
     * Significant changes:
     * - STILL to RESTLESS (agitation)
     * - RESTLESS to STILL (calming down)
     * - Multiple level jumps
     */
    private fun isSignificantActivityChange(
        from: NursingLabels.Activity,
        to: NursingLabels.Activity
    ): Boolean {
        // Restless is always significant
        if (from == NursingLabels.Activity.RESTLESS || to == NursingLabels.Activity.RESTLESS) {
            return true
        }

        // Calculate the "distance" between activity levels
        val levels = listOf(
            NursingLabels.Activity.STILL,
            NursingLabels.Activity.MINIMAL,
            NursingLabels.Activity.MODERATE,
            NursingLabels.Activity.ACTIVE,
            NursingLabels.Activity.RESTLESS
        )

        val fromIndex = levels.indexOf(from)
        val toIndex = levels.indexOf(to)

        // Significant if jump of 2+ levels
        return kotlin.math.abs(toIndex - fromIndex) >= 2
    }

    /**
     * Get statistics about scheduler performance
     */
    fun getStats(): SchedulerStats {
        return SchedulerStats(
            lastVlmTriggerTime = lastVlmTriggerTime,
            timeSinceLastVlm = if (lastVlmTriggerTime > 0)
                System.currentTimeMillis() - lastVlmTriggerTime else -1,
            currentPosition = previousPosition,
            currentAlertness = previousAlertness,
            currentActivity = previousActivity
        )
    }

    data class SchedulerStats(
        val lastVlmTriggerTime: Long,
        val timeSinceLastVlm: Long,
        val currentPosition: NursingLabels.Position?,
        val currentAlertness: NursingLabels.Alertness?,
        val currentActivity: NursingLabels.Activity?
    )
}
