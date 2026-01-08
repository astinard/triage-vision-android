package com.triage.vision.data

import com.triage.vision.charting.AutoChartEngine
import com.triage.vision.pipeline.SlowPipeline
import kotlinx.coroutines.flow.Flow
import java.util.concurrent.TimeUnit

class ObservationRepository(
    private val observationDao: ObservationDao
) {
    private val chartEngine = AutoChartEngine()

    /**
     * Get all observations as a Flow
     */
    fun getAllObservations(): Flow<List<ObservationEntity>> {
        return observationDao.getAllObservations()
    }

    /**
     * Get recent observations
     */
    fun getRecentObservations(limit: Int = 50): Flow<List<ObservationEntity>> {
        return observationDao.getRecentObservations(limit)
    }

    /**
     * Get observations from the last N hours
     */
    fun getObservationsFromLastHours(hours: Int): Flow<List<ObservationEntity>> {
        val since = System.currentTimeMillis() - TimeUnit.HOURS.toMillis(hours.toLong())
        return observationDao.getObservationsSince(since)
    }

    /**
     * Save a new observation
     */
    suspend fun saveObservation(
        observation: SlowPipeline.Observation,
        patientId: String? = null,
        triggeredBy: String = "interval",
        motionLevel: Float? = null,
        secondsStill: Long? = null
    ): ObservationEntity {
        // Generate FHIR JSON
        val chartEntry = chartEngine.generateChartEntry(
            observation = observation,
            patientId = patientId,
            motionHistory = motionLevel?.let {
                AutoChartEngine.MotionSummary(
                    averageMotionLevel = it,
                    secondsWithoutMotion = secondsStill ?: 0
                )
            }
        )
        val fhirJson = chartEngine.exportAsFHIRObservation(chartEntry)

        // Create entity
        val entity = ObservationEntity.fromObservation(
            observation = observation,
            patientId = patientId,
            triggeredBy = triggeredBy,
            motionLevel = motionLevel,
            secondsStill = secondsStill,
            fhirJson = fhirJson
        )

        // Save to database
        observationDao.insert(entity)

        return entity
    }

    /**
     * Get a specific observation by ID
     */
    suspend fun getObservation(id: String): ObservationEntity? {
        return observationDao.getById(id)
    }

    /**
     * Mark observation as exported
     */
    suspend fun markExported(id: String) {
        observationDao.markExported(id)
    }

    /**
     * Get unexported observations for batch export
     */
    suspend fun getUnexportedObservations(): List<ObservationEntity> {
        return observationDao.getUnexportedObservations()
    }

    /**
     * Delete old observations (for data retention compliance)
     */
    suspend fun purgeOldObservations(retentionHours: Int = 24) {
        val cutoff = System.currentTimeMillis() - TimeUnit.HOURS.toMillis(retentionHours.toLong())
        observationDao.deleteOlderThan(cutoff)
    }

    /**
     * Get observation count
     */
    suspend fun getObservationCount(): Int {
        return observationDao.getCount()
    }

    /**
     * Get observation count for today
     */
    suspend fun getTodayObservationCount(): Int {
        val startOfDay = java.time.LocalDate.now()
            .atStartOfDay(java.time.ZoneId.systemDefault())
            .toInstant()
            .toEpochMilli()
        return observationDao.getCountSince(startOfDay)
    }
}
