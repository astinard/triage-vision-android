package com.triage.vision.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.triage.vision.pipeline.SlowPipeline

@Entity(tableName = "observations")
data class ObservationEntity(
    @PrimaryKey
    val id: String,
    val timestamp: Long,
    val patientId: String?,

    // Observation fields
    val position: String,
    val alertness: String,
    val movementLevel: String,
    val equipmentVisible: List<String>,
    val concerns: List<String>,
    val comfortAssessment: String,
    val chartNote: String,

    // Motion context
    val averageMotionLevel: Float?,
    val secondsWithoutMotion: Long?,

    // Metadata
    val triggeredBy: String, // "interval", "stillness", "fall", "manual"
    val fhirJson: String?, // Exported FHIR resource

    // Sync status (for future EMR integration)
    val exported: Boolean = false,
    val exportedAt: Long? = null
) {
    companion object {
        fun fromObservation(
            observation: SlowPipeline.Observation,
            patientId: String? = null,
            triggeredBy: String = "interval",
            motionLevel: Float? = null,
            secondsStill: Long? = null,
            fhirJson: String? = null
        ): ObservationEntity {
            return ObservationEntity(
                id = java.util.UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                patientId = patientId,
                position = observation.position,
                alertness = observation.alertness,
                movementLevel = observation.movementLevel,
                equipmentVisible = observation.equipmentVisible,
                concerns = observation.concerns,
                comfortAssessment = observation.comfortAssessment,
                chartNote = observation.chartNote,
                averageMotionLevel = motionLevel,
                secondsWithoutMotion = secondsStill,
                triggeredBy = triggeredBy,
                fhirJson = fhirJson
            )
        }
    }

    fun toObservation(): SlowPipeline.Observation {
        return SlowPipeline.Observation(
            timestamp = java.time.Instant.ofEpochMilli(timestamp).toString(),
            position = position,
            alertness = alertness,
            movementLevel = movementLevel,
            equipmentVisible = equipmentVisible,
            concerns = concerns,
            comfortAssessment = comfortAssessment,
            chartNote = chartNote
        )
    }
}

@Entity(tableName = "alerts")
data class AlertEntity(
    @PrimaryKey
    val id: String = java.util.UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),
    val alertType: String, // "fall", "stillness", "pose_change"
    val details: String,
    val acknowledged: Boolean = false,
    val acknowledgedAt: Long? = null,
    val associatedObservationId: String? = null
)
