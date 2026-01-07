package com.triage.vision.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface ObservationDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(observation: ObservationEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(observations: List<ObservationEntity>)

    @Update
    suspend fun update(observation: ObservationEntity)

    @Delete
    suspend fun delete(observation: ObservationEntity)

    @Query("SELECT * FROM observations ORDER BY timestamp DESC")
    fun getAllObservations(): Flow<List<ObservationEntity>>

    @Query("SELECT * FROM observations ORDER BY timestamp DESC LIMIT :limit")
    fun getRecentObservations(limit: Int): Flow<List<ObservationEntity>>

    @Query("SELECT * FROM observations WHERE id = :id")
    suspend fun getById(id: String): ObservationEntity?

    @Query("SELECT * FROM observations WHERE patientId = :patientId ORDER BY timestamp DESC")
    fun getByPatientId(patientId: String): Flow<List<ObservationEntity>>

    @Query("SELECT * FROM observations WHERE timestamp >= :since ORDER BY timestamp DESC")
    fun getObservationsSince(since: Long): Flow<List<ObservationEntity>>

    @Query("SELECT * FROM observations WHERE exported = 0 ORDER BY timestamp ASC")
    suspend fun getUnexportedObservations(): List<ObservationEntity>

    @Query("UPDATE observations SET exported = 1, exportedAt = :exportedAt WHERE id = :id")
    suspend fun markExported(id: String, exportedAt: Long = System.currentTimeMillis())

    @Query("DELETE FROM observations WHERE timestamp < :before")
    suspend fun deleteOlderThan(before: Long)

    @Query("SELECT COUNT(*) FROM observations")
    suspend fun getCount(): Int

    @Query("SELECT COUNT(*) FROM observations WHERE timestamp >= :since")
    suspend fun getCountSince(since: Long): Int
}

@Dao
interface AlertDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(alert: AlertEntity)

    @Update
    suspend fun update(alert: AlertEntity)

    @Query("SELECT * FROM alerts ORDER BY timestamp DESC")
    fun getAllAlerts(): Flow<List<AlertEntity>>

    @Query("SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY timestamp DESC")
    fun getUnacknowledgedAlerts(): Flow<List<AlertEntity>>

    @Query("UPDATE alerts SET acknowledged = 1, acknowledgedAt = :acknowledgedAt WHERE id = :id")
    suspend fun acknowledge(id: String, acknowledgedAt: Long = System.currentTimeMillis())

    @Query("DELETE FROM alerts WHERE timestamp < :before")
    suspend fun deleteOlderThan(before: Long)
}
