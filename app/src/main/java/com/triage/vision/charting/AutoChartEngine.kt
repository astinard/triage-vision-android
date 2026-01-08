package com.triage.vision.charting

import com.triage.vision.pipeline.FastPipeline
import com.triage.vision.pipeline.SlowPipeline
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Auto-Charting Engine
 *
 * Converts VLM observations into structured nursing chart entries
 * compatible with HL7 FHIR R4 format for EMR integration.
 *
 * Supports:
 * - Individual Observation resources
 * - Bundle collections for batch export
 * - Alert/DetectedIssue resources
 * - Device provenance for automated observations
 * - Standard LOINC and SNOMED CT codes
 */
class AutoChartEngine {

    companion object {
        // LOINC codes for patient monitoring
        const val LOINC_BODY_POSITION = "8361-8"
        const val LOINC_CONSCIOUSNESS = "80288-4"
        const val LOINC_ACTIVITY_LEVEL = "62812-3"
        const val LOINC_MEDICAL_EQUIPMENT = "74468-0"
        const val LOINC_PATIENT_OBSERVATION = "75292-3"
        const val LOINC_FALL_RISK = "73830-0"
        const val LOINC_MOBILITY_STATUS = "88335-1"
        const val LOINC_PAIN_ASSESSMENT = "72514-3"

        // SNOMED CT codes
        const val SNOMED_SYSTEM = "http://snomed.info/sct"
        const val SNOMED_LYING = "102538003"      // Recumbent body position
        const val SNOMED_SITTING = "33586001"      // Sitting position
        const val SNOMED_STANDING = "10904000"     // Standing position
        const val SNOMED_ASLEEP = "248218005"      // Asleep
        const val SNOMED_AWAKE = "248220008"       // Awake
        const val SNOMED_FALL = "217082002"        // Fall

        // FHIR category codes
        const val CATEGORY_SURVEY = "survey"
        const val CATEGORY_VITAL_SIGNS = "vital-signs"
        const val CATEGORY_ACTIVITY = "activity"
    }

    private val json = Json {
        prettyPrint = true
        encodeDefaults = true
        ignoreUnknownKeys = true
    }

    private val compactJson = Json {
        prettyPrint = false
        encodeDefaults = false
        ignoreUnknownKeys = true
    }

    // ==================== FHIR R4 Resource Structures ====================

    /**
     * FHIR Reference to another resource
     */
    @Serializable
    data class Reference(
        val reference: String? = null,
        val display: String? = null,
        val type: String? = null
    )

    /**
     * FHIR Identifier
     */
    @Serializable
    data class Identifier(
        val system: String? = null,
        val value: String? = null
    )

    /**
     * FHIR Coding element
     */
    @Serializable
    data class Coding(
        val system: String = "",
        val code: String = "",
        val display: String = ""
    )

    /**
     * FHIR CodeableConcept
     */
    @Serializable
    data class CodeableConcept(
        val coding: List<Coding> = emptyList(),
        val text: String? = null
    )

    /**
     * FHIR Observation component
     */
    @Serializable
    data class ObservationComponent(
        val code: CodeableConcept,
        val valueString: String? = null,
        val valueCodeableConcept: CodeableConcept? = null,
        val valueQuantity: Quantity? = null
    )

    /**
     * FHIR Quantity
     */
    @Serializable
    data class Quantity(
        val value: Double? = null,
        val unit: String? = null,
        val system: String? = "http://unitsofmeasure.org",
        val code: String? = null
    )

    /**
     * HL7 FHIR R4 Observation resource
     */
    @Serializable
    data class FHIRObservation(
        val resourceType: String = "Observation",
        val id: String? = null,
        val identifier: List<Identifier>? = null,
        val status: String = "final",
        val category: List<CodeableConcept> = listOf(
            CodeableConcept(
                coding = listOf(
                    Coding(
                        system = "http://terminology.hl7.org/CodeSystem/observation-category",
                        code = CATEGORY_SURVEY,
                        display = "Survey"
                    )
                )
            )
        ),
        val code: CodeableConcept = CodeableConcept(
            coding = listOf(
                Coding(
                    system = "http://loinc.org",
                    code = LOINC_PATIENT_OBSERVATION,
                    display = "Patient monitoring observation"
                )
            )
        ),
        val subject: Reference? = null,
        val effectiveDateTime: String = "",
        val issued: String? = null,
        val performer: List<Reference>? = null,
        val device: Reference? = null,
        val valueString: String? = null,
        val valueCodeableConcept: CodeableConcept? = null,
        val note: List<Annotation>? = null,
        val component: List<ObservationComponent> = emptyList()
    )

    /**
     * FHIR Annotation (for notes)
     */
    @Serializable
    data class Annotation(
        val time: String? = null,
        val text: String
    )

    /**
     * FHIR DetectedIssue for alerts/concerns
     */
    @Serializable
    data class FHIRDetectedIssue(
        val resourceType: String = "DetectedIssue",
        val id: String? = null,
        val status: String = "final",
        val code: CodeableConcept? = null,
        val severity: String? = null,  // high | moderate | low
        val patient: Reference? = null,
        val identifiedDateTime: String? = null,
        val detail: String? = null,
        val reference: List<Reference>? = null
    )

    /**
     * FHIR Device resource for provenance
     */
    @Serializable
    data class FHIRDevice(
        val resourceType: String = "Device",
        val id: String = "triage-vision-device",
        val identifier: List<Identifier>? = listOf(
            Identifier(
                system = "urn:triage-vision:device",
                value = "triage-vision-android-1.0"
            )
        ),
        val deviceName: List<DeviceName>? = listOf(
            DeviceName(
                name = "Triage Vision Patient Monitor",
                type = "user-friendly-name"
            )
        ),
        val type: CodeableConcept? = CodeableConcept(
            coding = listOf(
                Coding(
                    system = "http://snomed.info/sct",
                    code = "706689003",
                    display = "Patient monitoring device"
                )
            )
        ),
        val version: List<DeviceVersion>? = listOf(
            DeviceVersion(value = "1.0.0")
        )
    )

    @Serializable
    data class DeviceName(
        val name: String,
        val type: String
    )

    @Serializable
    data class DeviceVersion(
        val value: String
    )

    /**
     * FHIR Bundle for collections
     */
    @Serializable
    data class FHIRBundle(
        val resourceType: String = "Bundle",
        val id: String? = null,
        val type: String = "collection",
        val timestamp: String? = null,
        val total: Int? = null,
        val entry: List<BundleEntry> = emptyList()
    )

    @Serializable
    data class BundleEntry(
        val fullUrl: String? = null,
        val resource: FHIRResource
    )

    /**
     * Union type for FHIR resources in a Bundle
     */
    @Serializable
    sealed class FHIRResource {
        @Serializable
        data class ObservationResource(val observation: FHIRObservation) : FHIRResource()

        @Serializable
        data class DetectedIssueResource(val detectedIssue: FHIRDetectedIssue) : FHIRResource()

        @Serializable
        data class DeviceResource(val device: FHIRDevice) : FHIRResource()
    }

    // ==================== Internal Data Structures ====================

    /**
     * Internal chart entry for storage
     */
    @Serializable
    data class ChartEntry(
        val id: String = java.util.UUID.randomUUID().toString(),
        val timestamp: String = Instant.now().toString(),
        val patientId: String? = null,
        val encounterId: String? = null,
        val observation: SlowPipeline.Observation,
        val motionHistory: MotionSummary? = null,
        val alerts: List<AlertInfo> = emptyList(),
        val fhirObservation: FHIRObservation? = null,
        val fhirAlerts: List<FHIRDetectedIssue> = emptyList()
    )

    @Serializable
    data class MotionSummary(
        val averageMotionLevel: Float = 0f,
        val secondsWithoutMotion: Long = 0,
        val poseChanges: Int = 0,
        val periodMinutes: Int = 15,
        val dominantPose: String = "unknown"
    )

    @Serializable
    data class AlertInfo(
        val type: String,
        val severity: AlertSeverity = AlertSeverity.MODERATE,
        val message: String,
        val timestamp: String = Instant.now().toString()
    )

    enum class AlertSeverity {
        LOW, MODERATE, HIGH
    }

    // ==================== Chart Entry Generation ====================

    /**
     * Generate chart entry from VLM observation
     */
    fun generateChartEntry(
        observation: SlowPipeline.Observation,
        patientId: String? = null,
        encounterId: String? = null,
        motionHistory: MotionSummary? = null,
        alerts: List<AlertInfo> = emptyList()
    ): ChartEntry {
        val entryId = java.util.UUID.randomUUID().toString()
        val timestamp = Instant.now().toString()

        // Build FHIR observation
        val fhirObservation = buildFHIRObservation(
            observation = observation,
            entryId = entryId,
            patientId = patientId,
            timestamp = timestamp,
            motionHistory = motionHistory
        )

        // Build FHIR alerts
        val fhirAlerts = alerts.map { alert ->
            buildFHIRDetectedIssue(
                alert = alert,
                patientId = patientId,
                observationRef = "Observation/$entryId"
            )
        }

        return ChartEntry(
            id = entryId,
            timestamp = timestamp,
            patientId = patientId,
            encounterId = encounterId,
            observation = observation,
            motionHistory = motionHistory,
            alerts = alerts,
            fhirObservation = fhirObservation,
            fhirAlerts = fhirAlerts
        )
    }

    /**
     * Generate chart entry from FastPipeline alert
     */
    fun generateAlertEntry(
        alert: FastPipeline.Alert,
        patientId: String? = null
    ): AlertInfo {
        val (type, severity, message) = when (alert) {
            is FastPipeline.Alert.FallDetected -> Triple(
                "fall_detected",
                AlertSeverity.HIGH,
                "Fall detected by motion analysis"
            )
            is FastPipeline.Alert.DepthVerifiedFall -> Triple(
                "fall_verified",
                AlertSeverity.HIGH,
                "Fall verified by depth sensor. Drop: ${String.format("%.2f", alert.dropMeters)}m, confidence: ${String.format("%.1f", alert.confidence * 100)}%"
            )
            is FastPipeline.Alert.LeavingBedZone -> Triple(
                "bed_exit",
                AlertSeverity.MODERATE,
                "Patient leaving bed zone. Distance: ${String.format("%.2f", alert.distanceMeters)}m"
            )
            is FastPipeline.Alert.Stillness -> Triple(
                "prolonged_stillness",
                AlertSeverity.MODERATE,
                "No movement detected for ${alert.durationSeconds / 60} minutes"
            )
            is FastPipeline.Alert.PoseChange -> Triple(
                "pose_change",
                AlertSeverity.LOW,
                "Position changed from ${alert.from.name.lowercase()} to ${alert.to.name.lowercase()}"
            )
        }

        return AlertInfo(
            type = type,
            severity = severity,
            message = message
        )
    }

    // ==================== FHIR Resource Builders ====================

    private fun buildFHIRObservation(
        observation: SlowPipeline.Observation,
        entryId: String,
        patientId: String?,
        timestamp: String,
        motionHistory: MotionSummary?
    ): FHIRObservation {
        val components = mutableListOf<ObservationComponent>()

        // Position component with SNOMED coding
        components.add(
            ObservationComponent(
                code = CodeableConcept(
                    coding = listOf(
                        Coding(
                            system = "http://loinc.org",
                            code = LOINC_BODY_POSITION,
                            display = "Body position"
                        )
                    )
                ),
                valueString = observation.position,
                valueCodeableConcept = mapPositionToSNOMED(observation.position)
            )
        )

        // Alertness component
        components.add(
            ObservationComponent(
                code = CodeableConcept(
                    coding = listOf(
                        Coding(
                            system = "http://loinc.org",
                            code = LOINC_CONSCIOUSNESS,
                            display = "Level of consciousness"
                        )
                    )
                ),
                valueString = observation.alertness,
                valueCodeableConcept = mapAlertnessToSNOMED(observation.alertness)
            )
        )

        // Activity level component
        components.add(
            ObservationComponent(
                code = CodeableConcept(
                    coding = listOf(
                        Coding(
                            system = "http://loinc.org",
                            code = LOINC_ACTIVITY_LEVEL,
                            display = "Physical activity level"
                        )
                    )
                ),
                valueString = observation.movementLevel
            )
        )

        // Mobility status from motion history
        motionHistory?.let { motion ->
            components.add(
                ObservationComponent(
                    code = CodeableConcept(
                        coding = listOf(
                            Coding(
                                system = "http://loinc.org",
                                code = LOINC_MOBILITY_STATUS,
                                display = "Mobility status"
                            )
                        )
                    ),
                    valueQuantity = Quantity(
                        value = motion.averageMotionLevel.toDouble(),
                        unit = "arbitrary unit",
                        code = "[arb'U]"
                    )
                )
            )

            // Stillness duration
            if (motion.secondsWithoutMotion > 0) {
                components.add(
                    ObservationComponent(
                        code = CodeableConcept(
                            coding = listOf(
                                Coding(
                                    system = "http://loinc.org",
                                    code = "68516-4",
                                    display = "Duration of inactivity"
                                )
                            )
                        ),
                        valueQuantity = Quantity(
                            value = motion.secondsWithoutMotion.toDouble(),
                            unit = "seconds",
                            code = "s"
                        )
                    )
                )
            }
        }

        // Equipment observed
        if (observation.equipmentVisible.isNotEmpty()) {
            components.add(
                ObservationComponent(
                    code = CodeableConcept(
                        coding = listOf(
                            Coding(
                                system = "http://loinc.org",
                                code = LOINC_MEDICAL_EQUIPMENT,
                                display = "Medical equipment"
                            )
                        )
                    ),
                    valueString = observation.equipmentVisible.joinToString(", ")
                )
            )
        }

        // Concerns as components
        if (observation.concerns.isNotEmpty() && observation.concerns != listOf("none")) {
            observation.concerns.forEach { concern ->
                components.add(
                    ObservationComponent(
                        code = CodeableConcept(
                            coding = listOf(
                                Coding(
                                    system = "http://loinc.org",
                                    code = "75326-9",
                                    display = "Problem"
                                )
                            ),
                            text = "Clinical concern"
                        ),
                        valueString = concern
                    )
                )
            }
        }

        return FHIRObservation(
            id = entryId,
            identifier = listOf(
                Identifier(
                    system = "urn:triage-vision:observation",
                    value = entryId
                )
            ),
            subject = patientId?.let {
                Reference(
                    reference = "Patient/$it",
                    type = "Patient"
                )
            },
            effectiveDateTime = timestamp,
            issued = timestamp,
            device = Reference(
                reference = "Device/triage-vision-device",
                display = "Triage Vision Patient Monitor"
            ),
            valueString = observation.chartNote,
            note = listOf(
                Annotation(
                    time = timestamp,
                    text = "Automated observation by Triage Vision AI monitoring system"
                )
            ),
            component = components
        )
    }

    private fun buildFHIRDetectedIssue(
        alert: AlertInfo,
        patientId: String?,
        observationRef: String?
    ): FHIRDetectedIssue {
        val issueCode = when (alert.type) {
            "fall_detected", "fall_verified" -> CodeableConcept(
                coding = listOf(
                    Coding(
                        system = SNOMED_SYSTEM,
                        code = SNOMED_FALL,
                        display = "Fall"
                    )
                ),
                text = "Fall detected"
            )
            "bed_exit" -> CodeableConcept(
                coding = listOf(
                    Coding(
                        system = SNOMED_SYSTEM,
                        code = "129839007",
                        display = "At risk for falls"
                    )
                ),
                text = "Patient leaving bed"
            )
            "prolonged_stillness" -> CodeableConcept(
                coding = listOf(
                    Coding(
                        system = SNOMED_SYSTEM,
                        code = "102891000",
                        display = "Age-related physical debility"
                    )
                ),
                text = "Prolonged inactivity"
            )
            else -> CodeableConcept(
                text = alert.type.replace("_", " ").replaceFirstChar { it.uppercase() }
            )
        }

        return FHIRDetectedIssue(
            id = java.util.UUID.randomUUID().toString(),
            status = "final",
            code = issueCode,
            severity = when (alert.severity) {
                AlertSeverity.HIGH -> "high"
                AlertSeverity.MODERATE -> "moderate"
                AlertSeverity.LOW -> "low"
            },
            patient = patientId?.let {
                Reference(reference = "Patient/$it", type = "Patient")
            },
            identifiedDateTime = alert.timestamp,
            detail = alert.message,
            reference = observationRef?.let { listOf(Reference(reference = it)) }
        )
    }

    private fun mapPositionToSNOMED(position: String): CodeableConcept? {
        val (code, display) = when (position.lowercase()) {
            "lying", "lying_in_bed", "supine" -> SNOMED_LYING to "Recumbent body position"
            "sitting", "seated" -> SNOMED_SITTING to "Sitting position"
            "standing" -> SNOMED_STANDING to "Standing position"
            else -> return null
        }
        return CodeableConcept(
            coding = listOf(Coding(system = SNOMED_SYSTEM, code = code, display = display))
        )
    }

    private fun mapAlertnessToSNOMED(alertness: String): CodeableConcept? {
        val (code, display) = when (alertness.lowercase()) {
            "asleep", "sleeping" -> SNOMED_ASLEEP to "Asleep"
            "awake", "alert" -> SNOMED_AWAKE to "Awake"
            "drowsy" -> "271782001" to "Drowsy"
            else -> return null
        }
        return CodeableConcept(
            coding = listOf(Coding(system = SNOMED_SYSTEM, code = code, display = display))
        )
    }

    // ==================== Export Methods ====================

    /**
     * Export chart entry as JSON string (internal format)
     */
    fun exportAsJson(entry: ChartEntry): String {
        return json.encodeToString(entry)
    }

    /**
     * Export chart entry as compact JSON (for transmission)
     */
    fun exportAsCompactJson(entry: ChartEntry): String {
        return compactJson.encodeToString(entry)
    }

    /**
     * Export only the FHIR Observation resource
     */
    fun exportAsFHIRObservation(entry: ChartEntry): String {
        return entry.fhirObservation?.let { json.encodeToString(it) } ?: "{}"
    }

    /**
     * Export all FHIR resources as a Bundle
     */
    fun exportAsFHIRBundle(entry: ChartEntry): String {
        val entries = mutableListOf<BundleEntry>()

        // Add observation
        entry.fhirObservation?.let { obs ->
            entries.add(
                BundleEntry(
                    fullUrl = "urn:uuid:${entry.id}",
                    resource = FHIRResource.ObservationResource(obs)
                )
            )
        }

        // Add alerts as DetectedIssue resources
        entry.fhirAlerts.forEach { issue ->
            entries.add(
                BundleEntry(
                    fullUrl = "urn:uuid:${issue.id}",
                    resource = FHIRResource.DetectedIssueResource(issue)
                )
            )
        }

        val bundle = FHIRBundle(
            id = java.util.UUID.randomUUID().toString(),
            type = "collection",
            timestamp = entry.timestamp,
            total = entries.size,
            entry = entries
        )

        return json.encodeToString(bundle)
    }

    /**
     * Export multiple chart entries as a FHIR Bundle
     */
    fun exportMultipleAsFHIRBundle(entries: List<ChartEntry>): String {
        val bundleEntries = mutableListOf<BundleEntry>()

        // Add device resource once
        bundleEntries.add(
            BundleEntry(
                fullUrl = "urn:uuid:device-triage-vision",
                resource = FHIRResource.DeviceResource(FHIRDevice())
            )
        )

        entries.forEach { entry ->
            entry.fhirObservation?.let { obs ->
                bundleEntries.add(
                    BundleEntry(
                        fullUrl = "urn:uuid:${entry.id}",
                        resource = FHIRResource.ObservationResource(obs)
                    )
                )
            }

            entry.fhirAlerts.forEach { issue ->
                bundleEntries.add(
                    BundleEntry(
                        fullUrl = "urn:uuid:${issue.id}",
                        resource = FHIRResource.DetectedIssueResource(issue)
                    )
                )
            }
        }

        val bundle = FHIRBundle(
            id = java.util.UUID.randomUUID().toString(),
            type = "collection",
            timestamp = Instant.now().toString(),
            total = bundleEntries.size,
            entry = bundleEntries
        )

        return json.encodeToString(bundle)
    }

    /**
     * Export as FHIR transaction bundle (for posting to FHIR server)
     */
    fun exportAsTransactionBundle(entries: List<ChartEntry>): String {
        val bundleEntries = mutableListOf<BundleEntry>()

        entries.forEach { entry ->
            entry.fhirObservation?.let { obs ->
                bundleEntries.add(
                    BundleEntry(
                        fullUrl = "urn:uuid:${entry.id}",
                        resource = FHIRResource.ObservationResource(obs)
                    )
                )
            }
        }

        val bundle = FHIRBundle(
            id = java.util.UUID.randomUUID().toString(),
            type = "transaction",
            timestamp = Instant.now().toString(),
            total = bundleEntries.size,
            entry = bundleEntries
        )

        return json.encodeToString(bundle)
    }

    /**
     * Get Device resource JSON
     */
    fun getDeviceResource(): String {
        return json.encodeToString(FHIRDevice())
    }

    // ==================== Human Readable Output ====================

    /**
     * Generate human-readable chart note
     */
    fun generateReadableNote(entry: ChartEntry): String {
        val obs = entry.observation
        val time = try {
            val instant = Instant.parse(entry.timestamp)
            DateTimeFormatter.ofPattern("HH:mm")
                .withZone(ZoneId.systemDefault())
                .format(instant)
        } catch (e: Exception) {
            "Unknown time"
        }

        return buildString {
            appendLine("[$time] Patient Monitoring Observation")
            appendLine("─".repeat(40))
            appendLine("Position:  ${formatField(obs.position)}")
            appendLine("Alertness: ${formatField(obs.alertness)}")
            appendLine("Activity:  ${formatField(obs.movementLevel)}")

            if (obs.equipmentVisible.isNotEmpty()) {
                appendLine("Equipment: ${obs.equipmentVisible.joinToString(", ") { formatField(it) }}")
            }

            entry.motionHistory?.let { motion ->
                appendLine()
                appendLine("Motion Summary (${motion.periodMinutes} min):")
                appendLine("  • Average motion: ${String.format("%.1f", motion.averageMotionLevel * 100)}%")
                if (motion.secondsWithoutMotion > 0) {
                    appendLine("  • Time without motion: ${formatDuration(motion.secondsWithoutMotion)}")
                }
                if (motion.poseChanges > 0) {
                    appendLine("  • Position changes: ${motion.poseChanges}")
                }
            }

            if (obs.concerns.isNotEmpty() && obs.concerns != listOf("none")) {
                appendLine()
                appendLine("⚠ Concerns:")
                obs.concerns.forEach { concern ->
                    appendLine("  • $concern")
                }
            }

            if (entry.alerts.isNotEmpty()) {
                appendLine()
                appendLine("🚨 Alerts:")
                entry.alerts.forEach { alert ->
                    val icon = when (alert.severity) {
                        AlertSeverity.HIGH -> "🔴"
                        AlertSeverity.MODERATE -> "🟡"
                        AlertSeverity.LOW -> "🟢"
                    }
                    appendLine("  $icon ${alert.message}")
                }
            }

            appendLine()
            appendLine("─".repeat(40))
            appendLine("Note: ${obs.chartNote}")
        }
    }

    /**
     * Generate brief summary for notifications
     */
    fun generateBriefSummary(entry: ChartEntry): String {
        val obs = entry.observation
        return buildString {
            append("${formatField(obs.position)}, ${formatField(obs.alertness)}")
            if (entry.alerts.isNotEmpty()) {
                append(" • ${entry.alerts.size} alert(s)")
            }
        }
    }

    private fun formatField(value: String): String {
        return value.replace("_", " ").replaceFirstChar { it.uppercase() }
    }

    private fun formatDuration(seconds: Long): String {
        return when {
            seconds < 60 -> "${seconds}s"
            seconds < 3600 -> "${seconds / 60}m ${seconds % 60}s"
            else -> "${seconds / 3600}h ${(seconds % 3600) / 60}m"
        }
    }

    // ==================== Validation ====================

    /**
     * Validate FHIR observation has required fields
     */
    fun validateFHIRObservation(observation: FHIRObservation): List<String> {
        val errors = mutableListOf<String>()

        if (observation.effectiveDateTime.isBlank()) {
            errors.add("Missing effectiveDateTime")
        }

        if (observation.code.coding.isEmpty()) {
            errors.add("Missing observation code")
        }

        if (observation.component.isEmpty()) {
            errors.add("No observation components")
        }

        return errors
    }

    /**
     * Check if entry has high-priority alerts
     */
    fun hasHighPriorityAlerts(entry: ChartEntry): Boolean {
        return entry.alerts.any { it.severity == AlertSeverity.HIGH }
    }
}
