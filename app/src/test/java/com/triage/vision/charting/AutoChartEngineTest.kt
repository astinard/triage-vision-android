package com.triage.vision.charting

import com.triage.vision.pipeline.FastPipeline
import com.triage.vision.pipeline.SlowPipeline
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for AutoChartEngine.
 *
 * Tests FHIR R4 compliance and charting functionality.
 */
class AutoChartEngineTest {

    private lateinit var chartEngine: AutoChartEngine
    private val json = Json { ignoreUnknownKeys = true }

    @Before
    fun setUp() {
        chartEngine = AutoChartEngine()
    }

    // ==================== FHIR Constants Tests ====================

    @Test
    fun `LOINC codes are valid format`() {
        // LOINC codes should be numeric with optional dash
        val loincPattern = Regex("^\\d+(-\\d+)?$")

        assertTrue(loincPattern.matches(AutoChartEngine.LOINC_BODY_POSITION))
        assertTrue(loincPattern.matches(AutoChartEngine.LOINC_CONSCIOUSNESS))
        assertTrue(loincPattern.matches(AutoChartEngine.LOINC_ACTIVITY_LEVEL))
        assertTrue(loincPattern.matches(AutoChartEngine.LOINC_MEDICAL_EQUIPMENT))
        assertTrue(loincPattern.matches(AutoChartEngine.LOINC_PATIENT_OBSERVATION))
        assertTrue(loincPattern.matches(AutoChartEngine.LOINC_FALL_RISK))
        assertTrue(loincPattern.matches(AutoChartEngine.LOINC_MOBILITY_STATUS))
        assertTrue(loincPattern.matches(AutoChartEngine.LOINC_PAIN_ASSESSMENT))
    }

    @Test
    fun `SNOMED codes are valid format`() {
        // SNOMED codes are numeric
        val snomedPattern = Regex("^\\d+$")

        assertTrue(snomedPattern.matches(AutoChartEngine.SNOMED_LYING))
        assertTrue(snomedPattern.matches(AutoChartEngine.SNOMED_SITTING))
        assertTrue(snomedPattern.matches(AutoChartEngine.SNOMED_STANDING))
        assertTrue(snomedPattern.matches(AutoChartEngine.SNOMED_ASLEEP))
        assertTrue(snomedPattern.matches(AutoChartEngine.SNOMED_AWAKE))
        assertTrue(snomedPattern.matches(AutoChartEngine.SNOMED_FALL))
    }

    @Test
    fun `SNOMED system URL is valid`() {
        assertEquals("http://snomed.info/sct", AutoChartEngine.SNOMED_SYSTEM)
    }

    // ==================== Chart Entry Generation Tests ====================

    @Test
    fun `generateChartEntry creates valid entry`() {
        val observation = createTestObservation()

        val entry = chartEngine.generateChartEntry(
            observation = observation,
            patientId = "patient-123",
            encounterId = "encounter-456"
        )

        assertNotNull(entry.id)
        assertTrue(entry.id.isNotBlank())
        assertNotNull(entry.timestamp)
        assertEquals("patient-123", entry.patientId)
        assertEquals("encounter-456", entry.encounterId)
        assertEquals(observation, entry.observation)
        assertNotNull(entry.fhirObservation)
    }

    @Test
    fun `generateChartEntry includes motion history`() {
        val observation = createTestObservation()
        val motionHistory = AutoChartEngine.MotionSummary(
            averageMotionLevel = 0.3f,
            secondsWithoutMotion = 120,
            poseChanges = 5,
            periodMinutes = 15,
            dominantPose = "lying_supine"
        )

        val entry = chartEngine.generateChartEntry(
            observation = observation,
            motionHistory = motionHistory
        )

        assertEquals(motionHistory, entry.motionHistory)
    }

    @Test
    fun `generateChartEntry includes alerts`() {
        val observation = createTestObservation()
        val alerts = listOf(
            AutoChartEngine.AlertInfo(
                type = "fall_detected",
                severity = AutoChartEngine.AlertSeverity.HIGH,
                message = "Fall detected"
            )
        )

        val entry = chartEngine.generateChartEntry(
            observation = observation,
            alerts = alerts
        )

        assertEquals(1, entry.alerts.size)
        assertEquals(1, entry.fhirAlerts.size)
        assertEquals("fall_detected", entry.alerts[0].type)
    }

    // ==================== Alert Generation Tests ====================

    @Test
    fun `generateAlertEntry for FallDetected`() {
        val alert = FastPipeline.Alert.FallDetected

        val alertInfo = chartEngine.generateAlertEntry(alert)

        assertEquals("fall_detected", alertInfo.type)
        assertEquals(AutoChartEngine.AlertSeverity.HIGH, alertInfo.severity)
        assertTrue(alertInfo.message.contains("Fall"))
    }

    @Test
    fun `generateAlertEntry for DepthVerifiedFall`() {
        val alert = FastPipeline.Alert.DepthVerifiedFall(
            dropMeters = 0.75f,
            confidence = 0.95f
        )

        val alertInfo = chartEngine.generateAlertEntry(alert)

        assertEquals("fall_verified", alertInfo.type)
        assertEquals(AutoChartEngine.AlertSeverity.HIGH, alertInfo.severity)
        assertTrue(alertInfo.message.contains("0.75"))
        assertTrue(alertInfo.message.contains("95"))
    }

    @Test
    fun `generateAlertEntry for LeavingBedZone`() {
        val alert = FastPipeline.Alert.LeavingBedZone(
            distanceMeters = 1.5f
        )

        val alertInfo = chartEngine.generateAlertEntry(alert)

        assertEquals("bed_exit", alertInfo.type)
        assertEquals(AutoChartEngine.AlertSeverity.MODERATE, alertInfo.severity)
    }

    @Test
    fun `generateAlertEntry for Stillness`() {
        val alert = FastPipeline.Alert.Stillness(
            durationSeconds = 1800  // 30 minutes
        )

        val alertInfo = chartEngine.generateAlertEntry(alert)

        assertEquals("prolonged_stillness", alertInfo.type)
        assertEquals(AutoChartEngine.AlertSeverity.MODERATE, alertInfo.severity)
        assertTrue(alertInfo.message.contains("30"))
    }

    @Test
    fun `generateAlertEntry for PoseChange`() {
        val alert = FastPipeline.Alert.PoseChange(
            from = FastPipeline.Pose.LYING,
            to = FastPipeline.Pose.SITTING
        )

        val alertInfo = chartEngine.generateAlertEntry(alert)

        assertEquals("pose_change", alertInfo.type)
        assertEquals(AutoChartEngine.AlertSeverity.LOW, alertInfo.severity)
        assertTrue(alertInfo.message.contains("lying"))
        assertTrue(alertInfo.message.contains("sitting"))
    }

    // ==================== FHIR Observation Tests ====================

    @Test
    fun `FHIR observation has required fields`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(observation = observation)

        val fhir = entry.fhirObservation!!

        assertEquals("Observation", fhir.resourceType)
        assertNotNull(fhir.id)
        assertEquals("final", fhir.status)
        assertTrue(fhir.effectiveDateTime.isNotBlank())
        assertTrue(fhir.code.coding.isNotEmpty())
        assertTrue(fhir.component.isNotEmpty())
    }

    @Test
    fun `FHIR observation has LOINC code`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(observation = observation)

        val coding = entry.fhirObservation!!.code.coding[0]

        assertEquals("http://loinc.org", coding.system)
        assertEquals(AutoChartEngine.LOINC_PATIENT_OBSERVATION, coding.code)
    }

    @Test
    fun `FHIR observation has device reference`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(observation = observation)

        val device = entry.fhirObservation!!.device

        assertNotNull(device)
        assertEquals("Device/triage-vision-device", device?.reference)
    }

    @Test
    fun `FHIR observation has patient reference when provided`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(
            observation = observation,
            patientId = "patient-123"
        )

        val subject = entry.fhirObservation!!.subject

        assertNotNull(subject)
        assertEquals("Patient/patient-123", subject?.reference)
        assertEquals("Patient", subject?.type)
    }

    @Test
    fun `FHIR observation has position component`() {
        val observation = createTestObservation(position = "sitting")
        val entry = chartEngine.generateChartEntry(observation = observation)

        val components = entry.fhirObservation!!.component
        val positionComponent = components.find {
            it.code.coding.any { coding -> coding.code == AutoChartEngine.LOINC_BODY_POSITION }
        }

        assertNotNull("Position component should exist", positionComponent)
        assertEquals("sitting", positionComponent?.valueString)
    }

    @Test
    fun `FHIR observation has alertness component`() {
        val observation = createTestObservation(alertness = "awake")
        val entry = chartEngine.generateChartEntry(observation = observation)

        val components = entry.fhirObservation!!.component
        val alertnessComponent = components.find {
            it.code.coding.any { coding -> coding.code == AutoChartEngine.LOINC_CONSCIOUSNESS }
        }

        assertNotNull("Alertness component should exist", alertnessComponent)
        assertEquals("awake", alertnessComponent?.valueString)
    }

    @Test
    fun `FHIR observation has activity component`() {
        val observation = createTestObservation(movementLevel = "moderate")
        val entry = chartEngine.generateChartEntry(observation = observation)

        val components = entry.fhirObservation!!.component
        val activityComponent = components.find {
            it.code.coding.any { coding -> coding.code == AutoChartEngine.LOINC_ACTIVITY_LEVEL }
        }

        assertNotNull("Activity component should exist", activityComponent)
        assertEquals("moderate", activityComponent?.valueString)
    }

    // ==================== FHIR Validation Tests ====================

    @Test
    fun `validateFHIRObservation passes for valid observation`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(observation = observation)

        val errors = chartEngine.validateFHIRObservation(entry.fhirObservation!!)

        assertTrue("Valid observation should have no errors", errors.isEmpty())
    }

    @Test
    fun `validateFHIRObservation fails for missing effectiveDateTime`() {
        val invalidObservation = AutoChartEngine.FHIRObservation(
            effectiveDateTime = "",  // Invalid - blank
            code = AutoChartEngine.CodeableConcept(
                coding = listOf(AutoChartEngine.Coding(
                    system = "http://loinc.org",
                    code = "75292-3"
                ))
            ),
            component = listOf(
                AutoChartEngine.ObservationComponent(
                    code = AutoChartEngine.CodeableConcept(),
                    valueString = "test"
                )
            )
        )

        val errors = chartEngine.validateFHIRObservation(invalidObservation)

        assertTrue("Should report missing effectiveDateTime",
            errors.any { it.contains("effectiveDateTime") })
    }

    @Test
    fun `validateFHIRObservation fails for missing code`() {
        val invalidObservation = AutoChartEngine.FHIRObservation(
            effectiveDateTime = "2024-01-01T00:00:00Z",
            code = AutoChartEngine.CodeableConcept(coding = emptyList()),  // Invalid
            component = listOf(
                AutoChartEngine.ObservationComponent(
                    code = AutoChartEngine.CodeableConcept(),
                    valueString = "test"
                )
            )
        )

        val errors = chartEngine.validateFHIRObservation(invalidObservation)

        assertTrue("Should report missing code", errors.any { it.contains("code") })
    }

    @Test
    fun `validateFHIRObservation fails for no components`() {
        val invalidObservation = AutoChartEngine.FHIRObservation(
            effectiveDateTime = "2024-01-01T00:00:00Z",
            code = AutoChartEngine.CodeableConcept(
                coding = listOf(AutoChartEngine.Coding(
                    system = "http://loinc.org",
                    code = "75292-3"
                ))
            ),
            component = emptyList()  // Invalid
        )

        val errors = chartEngine.validateFHIRObservation(invalidObservation)

        assertTrue("Should report no components", errors.any { it.contains("component") })
    }

    // ==================== FHIR DetectedIssue Tests ====================

    @Test
    fun `FHIR DetectedIssue has correct severity mapping`() {
        val observation = createTestObservation()
        val alerts = listOf(
            AutoChartEngine.AlertInfo(
                type = "fall_detected",
                severity = AutoChartEngine.AlertSeverity.HIGH,
                message = "Fall detected"
            )
        )

        val entry = chartEngine.generateChartEntry(observation = observation, alerts = alerts)

        val issue = entry.fhirAlerts[0]
        assertEquals("high", issue.severity)
    }

    @Test
    fun `FHIR DetectedIssue has SNOMED code for fall`() {
        val observation = createTestObservation()
        val alerts = listOf(
            AutoChartEngine.AlertInfo(
                type = "fall_detected",
                severity = AutoChartEngine.AlertSeverity.HIGH,
                message = "Fall detected"
            )
        )

        val entry = chartEngine.generateChartEntry(observation = observation, alerts = alerts)

        val issue = entry.fhirAlerts[0]
        val coding = issue.code?.coding?.get(0)
        assertEquals(AutoChartEngine.SNOMED_SYSTEM, coding?.system)
        assertEquals(AutoChartEngine.SNOMED_FALL, coding?.code)
    }

    // ==================== Export Tests ====================

    @Test
    fun `exportAsJson produces valid JSON`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(observation = observation)

        val jsonString = chartEngine.exportAsJson(entry)

        // Should be parseable JSON
        assertDoesNotThrow { json.parseToJsonElement(jsonString) }
        assertTrue(jsonString.contains("\"timestamp\""))
        assertTrue(jsonString.contains("\"observation\""))
    }

    @Test
    fun `exportAsFHIRObservation produces valid FHIR JSON`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(observation = observation)

        val jsonString = chartEngine.exportAsFHIRObservation(entry)
        val jsonObj = json.parseToJsonElement(jsonString).jsonObject

        assertEquals("Observation", jsonObj["resourceType"]?.jsonPrimitive?.content)
        assertEquals("final", jsonObj["status"]?.jsonPrimitive?.content)
    }

    @Test
    fun `exportAsFHIRBundle produces valid Bundle`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(observation = observation)

        val jsonString = chartEngine.exportAsFHIRBundle(entry)
        val jsonObj = json.parseToJsonElement(jsonString).jsonObject

        assertEquals("Bundle", jsonObj["resourceType"]?.jsonPrimitive?.content)
        assertEquals("collection", jsonObj["type"]?.jsonPrimitive?.content)
        assertTrue(jsonObj.containsKey("entry"))
    }

    @Test
    fun `exportMultipleAsFHIRBundle includes device resource`() {
        val entries = listOf(
            chartEngine.generateChartEntry(createTestObservation(position = "lying")),
            chartEngine.generateChartEntry(createTestObservation(position = "sitting"))
        )

        val jsonString = chartEngine.exportMultipleAsFHIRBundle(entries)
        val jsonObj = json.parseToJsonElement(jsonString).jsonObject

        // Should have device + 2 observations
        val entryArray = jsonObj["entry"]?.jsonArray
        assertNotNull(entryArray)
        assertTrue("Should have at least 3 entries (device + 2 obs)", (entryArray?.size ?: 0) >= 3)
    }

    @Test
    fun `getDeviceResource returns valid device JSON`() {
        val jsonString = chartEngine.getDeviceResource()
        val jsonObj = json.parseToJsonElement(jsonString).jsonObject

        assertEquals("Device", jsonObj["resourceType"]?.jsonPrimitive?.content)
        assertEquals("triage-vision-device", jsonObj["id"]?.jsonPrimitive?.content)
    }

    // ==================== Human Readable Output Tests ====================

    @Test
    fun `generateReadableNote includes all sections`() {
        val observation = createTestObservation(
            position = "lying_supine",
            alertness = "sleeping",
            movementLevel = "still",
            chartNote = "Patient is resting comfortably."
        )
        val entry = chartEngine.generateChartEntry(observation = observation)

        val note = chartEngine.generateReadableNote(entry)

        assertTrue(note.contains("Position:"))
        assertTrue(note.contains("Alertness:"))
        assertTrue(note.contains("Activity:"))
        assertTrue(note.contains("Patient is resting comfortably"))
    }

    @Test
    fun `generateReadableNote includes motion summary`() {
        val observation = createTestObservation()
        val motionHistory = AutoChartEngine.MotionSummary(
            averageMotionLevel = 0.2f,
            secondsWithoutMotion = 3600,  // 1 hour
            poseChanges = 3,
            periodMinutes = 15
        )
        val entry = chartEngine.generateChartEntry(
            observation = observation,
            motionHistory = motionHistory
        )

        val note = chartEngine.generateReadableNote(entry)

        assertTrue(note.contains("Motion Summary"))
        assertTrue(note.contains("20.0%"))  // 0.2 * 100
        assertTrue(note.contains("Position changes: 3"))
    }

    @Test
    fun `generateReadableNote includes alerts with icons`() {
        val observation = createTestObservation()
        val alerts = listOf(
            AutoChartEngine.AlertInfo(
                type = "fall_detected",
                severity = AutoChartEngine.AlertSeverity.HIGH,
                message = "Fall detected"
            )
        )
        val entry = chartEngine.generateChartEntry(observation = observation, alerts = alerts)

        val note = chartEngine.generateReadableNote(entry)

        assertTrue(note.contains("Alerts:"))
        assertTrue(note.contains("Fall detected"))
    }

    @Test
    fun `generateBriefSummary is concise`() {
        val observation = createTestObservation(
            position = "sitting",
            alertness = "awake"
        )
        val entry = chartEngine.generateChartEntry(observation = observation)

        val summary = chartEngine.generateBriefSummary(entry)

        assertTrue(summary.length < 100)  // Should be brief
        assertTrue(summary.contains("Sitting") || summary.contains("sitting"))
        assertTrue(summary.contains("Awake") || summary.contains("awake"))
    }

    // ==================== Priority Alert Tests ====================

    @Test
    fun `hasHighPriorityAlerts returns true for HIGH severity`() {
        val observation = createTestObservation()
        val alerts = listOf(
            AutoChartEngine.AlertInfo(
                type = "fall_detected",
                severity = AutoChartEngine.AlertSeverity.HIGH,
                message = "Fall detected"
            )
        )
        val entry = chartEngine.generateChartEntry(observation = observation, alerts = alerts)

        assertTrue(chartEngine.hasHighPriorityAlerts(entry))
    }

    @Test
    fun `hasHighPriorityAlerts returns false for only moderate alerts`() {
        val observation = createTestObservation()
        val alerts = listOf(
            AutoChartEngine.AlertInfo(
                type = "bed_exit",
                severity = AutoChartEngine.AlertSeverity.MODERATE,
                message = "Patient leaving bed"
            )
        )
        val entry = chartEngine.generateChartEntry(observation = observation, alerts = alerts)

        assertFalse(chartEngine.hasHighPriorityAlerts(entry))
    }

    @Test
    fun `hasHighPriorityAlerts returns false for no alerts`() {
        val observation = createTestObservation()
        val entry = chartEngine.generateChartEntry(observation = observation)

        assertFalse(chartEngine.hasHighPriorityAlerts(entry))
    }

    // ==================== Helper Methods ====================

    private fun createTestObservation(
        position: String = "lying",
        alertness: String = "sleeping",
        movementLevel: String = "still",
        equipmentVisible: List<String> = listOf("IV_pole", "bed_rails"),
        concerns: List<String> = listOf("none"),
        chartNote: String = "Patient appears comfortable and resting."
    ): SlowPipeline.Observation {
        return SlowPipeline.Observation(
            timestamp = java.time.Instant.now().toString(),
            position = position,
            alertness = alertness,
            movementLevel = movementLevel,
            equipmentVisible = equipmentVisible,
            concerns = concerns,
            comfortAssessment = "comfortable",
            chartNote = chartNote
        )
    }

    /**
     * Helper to assert a block doesn't throw
     */
    private inline fun assertDoesNotThrow(block: () -> Unit) {
        try {
            block()
        } catch (e: Exception) {
            fail("Expected no exception, but got: ${e.message}")
        }
    }
}
