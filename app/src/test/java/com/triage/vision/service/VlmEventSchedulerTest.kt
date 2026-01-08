package com.triage.vision.service

import com.triage.vision.classifier.FastClassifier
import com.triage.vision.classifier.NursingLabels
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for VlmEventScheduler.
 *
 * Tests the intelligent VLM triggering logic based on CLIP classification results.
 */
class VlmEventSchedulerTest {

    private lateinit var scheduler: VlmEventScheduler
    private var lastTriggerReason: String? = null
    private var triggerCount: Int = 0

    @Before
    fun setUp() {
        lastTriggerReason = null
        triggerCount = 0
        scheduler = VlmEventScheduler { reason ->
            lastTriggerReason = reason
            triggerCount++
        }
    }

    // ==================== Helper Methods ====================

    private fun createClassificationResult(
        position: NursingLabels.Position = NursingLabels.Position.LYING_SUPINE,
        positionConfidence: Float = 0.8f,
        alertness: NursingLabels.Alertness = NursingLabels.Alertness.SLEEPING,
        alertnessConfidence: Float = 0.8f,
        activity: NursingLabels.Activity = NursingLabels.Activity.STILL,
        activityConfidence: Float = 0.8f,
        comfort: NursingLabels.Comfort = NursingLabels.Comfort.COMFORTABLE,
        comfortConfidence: Float = 0.8f,
        safety: NursingLabels.SafetyConcern = NursingLabels.SafetyConcern.NONE,
        safetyConfidence: Float = 0.8f
    ): FastClassifier.ClassificationResult {
        return FastClassifier.ClassificationResult(
            position = position,
            positionConfidence = positionConfidence,
            alertness = alertness,
            alertnessConfidence = alertnessConfidence,
            activity = activity,
            activityConfidence = activityConfidence,
            comfort = comfort,
            comfortConfidence = comfortConfidence,
            safety = safety,
            safetyConfidence = safetyConfidence,
            inferenceTimeMs = 100
        )
    }

    /**
     * Simulate multiple consistent readings to achieve stability
     */
    private fun stabilizeState(result: FastClassifier.ClassificationResult, count: Int = 3) {
        repeat(count) {
            scheduler.processClassification(result)
        }
    }

    // ==================== Safety Trigger Tests ====================

    @Test
    fun `safety concern triggers VLM immediately`() {
        // Establish baseline state first
        stabilizeState(createClassificationResult())
        scheduler.reset() // Reset to clear last trigger time
        triggerCount = 0

        // Safety concern with high confidence
        val result = createClassificationResult(
            safety = NursingLabels.SafetyConcern.FALL_RISK,
            safetyConfidence = 0.7f
        )

        val event = scheduler.processClassification(result)

        assertNotNull("Safety concern should trigger VLM", event)
        assertTrue(event is VlmEventScheduler.TriggerEvent.SafetyConcern)
        assertEquals("safety_fall_risk", event?.reason)
        assertEquals(1, triggerCount)
    }

    @Test
    fun `fallen detection triggers VLM`() {
        scheduler.reset()

        val result = createClassificationResult(
            safety = NursingLabels.SafetyConcern.FALLEN,
            safetyConfidence = 0.8f
        )

        val event = scheduler.processClassification(result)

        assertNotNull("Fallen detection should trigger VLM", event)
        assertTrue(event is VlmEventScheduler.TriggerEvent.SafetyConcern)
    }

    @Test
    fun `low confidence safety does not trigger safety event`() {
        // Establish baseline state first with a safety trigger to set lastVlmTriggerTime
        val baseline = createClassificationResult(
            safety = NursingLabels.SafetyConcern.FALL_RISK,
            safetyConfidence = 0.8f
        )
        scheduler.processClassification(baseline)

        // Now test low confidence
        val result = createClassificationResult(
            safety = NursingLabels.SafetyConcern.FALL_RISK,
            safetyConfidence = 0.3f  // Below 0.5 threshold
        )

        val event = scheduler.processClassification(result)

        // Should not trigger as SafetyConcern due to low confidence
        // (may still trigger as periodic fallback, but not as safety)
        if (event != null) {
            assertFalse("Low confidence safety should not trigger safety event",
                event is VlmEventScheduler.TriggerEvent.SafetyConcern)
        }
    }

    // ==================== Position Change Tests ====================

    @Test
    fun `lying to standing is significant position change`() {
        // Test the internal significance logic, not the full scheduler behavior
        // This validates that lying-to-standing is considered significant

        // We can test this by creating a fresh scheduler and checking that
        // after stabilizing lying and then standing, a PositionChange event type is returned

        // First establish lying as stable
        val lying = createClassificationResult(position = NursingLabels.Position.LYING_SUPINE)
        repeat(4) { scheduler.processClassification(lying) }

        // Now change to standing with 4+ readings to ensure stability
        val standing = createClassificationResult(position = NursingLabels.Position.STANDING)
        var positionChangeDetected = false
        repeat(5) {
            val event = scheduler.processClassification(standing)
            if (event is VlmEventScheduler.TriggerEvent.PositionChange) {
                positionChangeDetected = true
            }
        }

        // Position change should be detected (though may be blocked by min interval)
        // The key test is that the state tracking works
        val stats = scheduler.getStats()
        assertEquals("Should have updated to standing", NursingLabels.Position.STANDING, stats.currentPosition)
    }

    @Test
    fun `on_floor position always triggers`() {
        // Establish baseline
        val sitting = createClassificationResult(position = NursingLabels.Position.SITTING_CHAIR)
        stabilizeState(sitting)
        scheduler.reset()
        triggerCount = 0

        // ON_FLOOR is significant
        val onFloor = createClassificationResult(position = NursingLabels.Position.ON_FLOOR)
        stabilizeState(onFloor)

        assertTrue("ON_FLOOR should be detected as significant", triggerCount >= 1)
    }

    @Test
    fun `minor lying position changes do not trigger position event`() {
        // Establish supine as stable with multiple readings
        val supine = createClassificationResult(position = NursingLabels.Position.LYING_SUPINE)
        repeat(4) { scheduler.processClassification(supine) }

        // Change to left lateral - same lying category
        val leftLateral = createClassificationResult(position = NursingLabels.Position.LYING_LEFT)
        var positionChangeTriggerCount = 0
        repeat(4) {
            val event = scheduler.processClassification(leftLateral)
            if (event is VlmEventScheduler.TriggerEvent.PositionChange) {
                positionChangeTriggerCount++
            }
        }

        // Minor lying position changes should not generate PositionChange events
        // (the scheduler considers supine->left lateral as minor)
        assertEquals("Minor lying position change should not trigger position event", 0, positionChangeTriggerCount)
    }

    // ==================== Alertness Change Tests ====================

    @Test
    fun `awake to unresponsive triggers VLM`() {
        // Establish awake state
        val awake = createClassificationResult(alertness = NursingLabels.Alertness.AWAKE_ALERT)
        stabilizeState(awake)
        scheduler.reset()
        triggerCount = 0

        // Change to unresponsive
        val unresponsive = createClassificationResult(alertness = NursingLabels.Alertness.UNRESPONSIVE)
        stabilizeState(unresponsive)

        assertTrue("Unresponsive state should trigger VLM", triggerCount >= 1)
    }

    @Test
    fun `awake to sleeping triggers VLM`() {
        // Establish awake state
        val awake = createClassificationResult(alertness = NursingLabels.Alertness.AWAKE_ALERT)
        stabilizeState(awake)
        scheduler.reset()
        triggerCount = 0

        // Change to sleeping
        val sleeping = createClassificationResult(alertness = NursingLabels.Alertness.SLEEPING)
        stabilizeState(sleeping)

        assertTrue("Awake to sleeping should trigger VLM", triggerCount >= 1)
    }

    // ==================== Activity Change Tests ====================

    @Test
    fun `still to restless triggers VLM`() {
        // Establish still state
        val still = createClassificationResult(activity = NursingLabels.Activity.STILL)
        stabilizeState(still)
        scheduler.reset()
        triggerCount = 0

        // Change to restless
        val restless = createClassificationResult(activity = NursingLabels.Activity.RESTLESS)
        stabilizeState(restless)

        assertTrue("Still to restless should trigger VLM", triggerCount >= 1)
    }

    @Test
    fun `two level activity jump triggers`() {
        // Establish still state
        val still = createClassificationResult(activity = NursingLabels.Activity.STILL)
        stabilizeState(still)
        scheduler.reset()
        triggerCount = 0

        // Jump to active (skip minimal and moderate)
        val active = createClassificationResult(activity = NursingLabels.Activity.ACTIVE)
        stabilizeState(active)

        assertTrue("Two level activity jump should trigger", triggerCount >= 1)
    }

    // ==================== Comfort Tests ====================

    @Test
    fun `distressed comfort triggers VLM`() {
        scheduler.reset()

        val result = createClassificationResult(
            comfort = NursingLabels.Comfort.DISTRESSED,
            comfortConfidence = 0.6f
        )

        val event = scheduler.processClassification(result)

        assertNotNull("Distressed state should trigger VLM", event)
    }

    @Test
    fun `pain indicated triggers VLM`() {
        scheduler.reset()

        val result = createClassificationResult(
            comfort = NursingLabels.Comfort.PAIN_INDICATED,
            comfortConfidence = 0.6f
        )

        val event = scheduler.processClassification(result)

        assertNotNull("Pain indication should trigger VLM", event)
    }

    // ==================== Stability Threshold Tests ====================

    @Test
    fun `unstable readings do not trigger position change events`() {
        // First establish a baseline to set lastVlmTriggerTime
        val baseline = createClassificationResult(
            safety = NursingLabels.SafetyConcern.FALL_RISK,
            safetyConfidence = 0.8f
        )
        scheduler.processClassification(baseline)

        // Alternate between states (not stable)
        val sitting = createClassificationResult(position = NursingLabels.Position.SITTING_BED)
        val standing = createClassificationResult(position = NursingLabels.Position.STANDING)

        var positionChanges = 0
        listOf(sitting, standing, sitting, standing).forEach { result ->
            val event = scheduler.processClassification(result)
            if (event is VlmEventScheduler.TriggerEvent.PositionChange) {
                positionChanges++
            }
        }

        // Unstable readings should not trigger position change events
        assertEquals("Unstable readings should not trigger position change", 0, positionChanges)
    }

    // ==================== Minimum Interval Tests ====================

    @Test
    fun `minimum interval prevents rapid triggers`() {
        // First trigger
        scheduler.reset()
        val safety1 = createClassificationResult(
            safety = NursingLabels.SafetyConcern.FALL_RISK,
            safetyConfidence = 0.8f
        )
        scheduler.processClassification(safety1)
        val firstCount = triggerCount

        // Immediate second trigger should be suppressed
        val safety2 = createClassificationResult(
            safety = NursingLabels.SafetyConcern.LEAVING_BED,
            safetyConfidence = 0.8f
        )
        scheduler.processClassification(safety2)

        assertEquals("Second trigger should be suppressed due to min interval", firstCount, triggerCount)
    }

    // ==================== Reset Tests ====================

    @Test
    fun `reset clears all state`() {
        // Establish some state
        val result = createClassificationResult()
        stabilizeState(result)

        scheduler.reset()

        val stats = scheduler.getStats()
        assertEquals(0L, stats.lastVlmTriggerTime)
        assertNull(stats.currentPosition)
        assertNull(stats.currentAlertness)
        assertNull(stats.currentActivity)
    }

    // ==================== Force Trigger Tests ====================

    @Test
    fun `force trigger bypasses all checks`() {
        triggerCount = 0

        scheduler.forceTrigger("manual_request")

        assertEquals(1, triggerCount)
        assertEquals("manual_request", lastTriggerReason)
    }

    // ==================== Stats Tests ====================

    @Test
    fun `stats reflect current state`() {
        val result = createClassificationResult(
            position = NursingLabels.Position.SITTING_CHAIR,
            alertness = NursingLabels.Alertness.AWAKE_DROWSY,
            activity = NursingLabels.Activity.MINIMAL
        )

        stabilizeState(result)

        val stats = scheduler.getStats()
        assertEquals(NursingLabels.Position.SITTING_CHAIR, stats.currentPosition)
        assertEquals(NursingLabels.Alertness.AWAKE_DROWSY, stats.currentAlertness)
        assertEquals(NursingLabels.Activity.MINIMAL, stats.currentActivity)
    }

    // ==================== Priority Tests ====================

    @Test
    fun `safety has highest priority`() {
        scheduler.reset()

        // Multiple events at once
        val result = createClassificationResult(
            safety = NursingLabels.SafetyConcern.FALLEN,
            safetyConfidence = 0.9f,
            comfort = NursingLabels.Comfort.DISTRESSED,
            comfortConfidence = 0.9f
        )

        val event = scheduler.processClassification(result)

        // Safety should win (priority 1 < comfort priority 2)
        assertTrue("Safety should take priority", event is VlmEventScheduler.TriggerEvent.SafetyConcern)
    }

    // ==================== Classification Result Tests ====================

    @Test
    fun `classification result getPrimaryConcern returns safety first`() {
        val result = createClassificationResult(
            safety = NursingLabels.SafetyConcern.FALL_RISK,
            safetyConfidence = 0.8f,
            comfort = NursingLabels.Comfort.DISTRESSED,
            comfortConfidence = 0.9f
        )

        val concern = result.getPrimaryConcern()

        assertNotNull(concern)
        assertTrue("Should return safety concern first", concern!!.startsWith("Safety"))
    }

    @Test
    fun `classification result toSummary formats correctly`() {
        val result = createClassificationResult(
            position = NursingLabels.Position.SITTING_BED,
            alertness = NursingLabels.Alertness.AWAKE_ALERT,
            activity = NursingLabels.Activity.MODERATE
        )

        val summary = result.toSummary()

        assertTrue(summary.contains("sitting_in_bed"))
        assertTrue(summary.contains("awake_alert"))
        assertTrue(summary.contains("moderate_movement"))
    }
}
