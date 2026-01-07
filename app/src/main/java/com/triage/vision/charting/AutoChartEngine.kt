package com.triage.vision.charting

import com.triage.vision.pipeline.FastPipeline
import com.triage.vision.pipeline.SlowPipeline
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.time.Instant
import java.time.format.DateTimeFormatter

/**
 * Auto-Charting Engine
 *
 * Converts VLM observations into structured nursing chart entries
 * compatible with HL7 FHIR format for EMR integration.
 */
class AutoChartEngine {

    private val json = Json {
        prettyPrint = true
        encodeDefaults = true
    }

    /**
     * HL7 FHIR Observation resource structure
     */
    @Serializable
    data class FHIRObservation(
        val resourceType: String = "Observation",
        val status: String = "final",
        val category: List<CodeableConcept> = listOf(
            CodeableConcept(
                coding = listOf(
                    Coding(
                        system = "http://terminology.hl7.org/CodeSystem/observation-category",
                        code = "survey",
                        display = "Survey"
                    )
                )
            )
        ),
        val code: CodeableConcept = CodeableConcept(
            coding = listOf(
                Coding(
                    system = "http://loinc.org",
                    code = "75292-3",
                    display = "Patient monitoring observation"
                )
            )
        ),
        val effectiveDateTime: String = "",
        val valueString: String = "",
        val component: List<ObservationComponent> = emptyList()
    )

    @Serializable
    data class CodeableConcept(
        val coding: List<Coding> = emptyList(),
        val text: String? = null
    )

    @Serializable
    data class Coding(
        val system: String = "",
        val code: String = "",
        val display: String = ""
    )

    @Serializable
    data class ObservationComponent(
        val code: CodeableConcept,
        val valueString: String? = null,
        val valueCodeableConcept: CodeableConcept? = null
    )

    /**
     * Internal chart entry for storage
     */
    @Serializable
    data class ChartEntry(
        val id: String = java.util.UUID.randomUUID().toString(),
        val timestamp: String = Instant.now().toString(),
        val patientId: String? = null,
        val observation: SlowPipeline.Observation,
        val motionHistory: MotionSummary? = null,
        val alerts: List<String> = emptyList(),
        val fhirResource: FHIRObservation? = null
    )

    @Serializable
    data class MotionSummary(
        val averageMotionLevel: Float = 0f,
        val secondsWithoutMotion: Long = 0,
        val poseChanges: Int = 0,
        val periodMinutes: Int = 15
    )

    /**
     * Generate chart entry from VLM observation
     */
    fun generateChartEntry(
        observation: SlowPipeline.Observation,
        patientId: String? = null,
        motionHistory: MotionSummary? = null,
        alerts: List<String> = emptyList()
    ): ChartEntry {
        val timestamp = Instant.now().toString()

        // Build FHIR observation
        val fhirObservation = buildFHIRObservation(observation, timestamp)

        return ChartEntry(
            timestamp = timestamp,
            patientId = patientId,
            observation = observation,
            motionHistory = motionHistory,
            alerts = alerts,
            fhirResource = fhirObservation
        )
    }

    private fun buildFHIRObservation(
        observation: SlowPipeline.Observation,
        timestamp: String
    ): FHIRObservation {
        val components = mutableListOf<ObservationComponent>()

        // Position component
        components.add(
            ObservationComponent(
                code = CodeableConcept(
                    coding = listOf(
                        Coding(
                            system = "http://loinc.org",
                            code = "8361-8",
                            display = "Body position"
                        )
                    )
                ),
                valueString = observation.position
            )
        )

        // Alertness component
        components.add(
            ObservationComponent(
                code = CodeableConcept(
                    coding = listOf(
                        Coding(
                            system = "http://loinc.org",
                            code = "80288-4",
                            display = "Level of consciousness"
                        )
                    )
                ),
                valueString = observation.alertness
            )
        )

        // Activity level component
        components.add(
            ObservationComponent(
                code = CodeableConcept(
                    coding = listOf(
                        Coding(
                            system = "http://loinc.org",
                            code = "62812-3",
                            display = "Physical activity level"
                        )
                    )
                ),
                valueString = observation.movementLevel
            )
        )

        // Equipment observed
        if (observation.equipmentVisible.isNotEmpty()) {
            components.add(
                ObservationComponent(
                    code = CodeableConcept(
                        coding = listOf(
                            Coding(
                                system = "http://loinc.org",
                                code = "74468-0",
                                display = "Medical equipment"
                            )
                        )
                    ),
                    valueString = observation.equipmentVisible.joinToString(", ")
                )
            )
        }

        return FHIRObservation(
            effectiveDateTime = timestamp,
            valueString = observation.chartNote,
            component = components
        )
    }

    /**
     * Export chart entry as JSON string
     */
    fun exportAsJson(entry: ChartEntry): String {
        return json.encodeToString(entry)
    }

    /**
     * Export only the FHIR resource
     */
    fun exportAsFHIR(entry: ChartEntry): String {
        return entry.fhirResource?.let { json.encodeToString(it) } ?: "{}"
    }

    /**
     * Generate human-readable chart note
     */
    fun generateReadableNote(entry: ChartEntry): String {
        val obs = entry.observation
        val time = try {
            val instant = Instant.parse(entry.timestamp)
            DateTimeFormatter.ofPattern("HH:mm")
                .withZone(java.time.ZoneId.systemDefault())
                .format(instant)
        } catch (e: Exception) {
            "Unknown time"
        }

        return buildString {
            appendLine("[$time] Patient Monitoring Observation")
            appendLine("Position: ${obs.position.replace("_", " ")}")
            appendLine("Alertness: ${obs.alertness.replace("_", " ")}")
            appendLine("Activity: ${obs.movementLevel.replace("_", " ")}")

            if (obs.equipmentVisible.isNotEmpty()) {
                appendLine("Equipment: ${obs.equipmentVisible.joinToString(", ") { it.replace("_", " ") }}")
            }

            if (obs.concerns.isNotEmpty() && obs.concerns != listOf("none")) {
                appendLine("Concerns: ${obs.concerns.joinToString("; ")}")
            }

            appendLine()
            appendLine("Note: ${obs.chartNote}")
        }
    }
}
