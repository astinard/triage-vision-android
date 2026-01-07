package com.triage.vision.camera

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs

/**
 * Synchronizes RGB and Depth frames by timestamp matching.
 *
 * RGB frames come from CameraX (nanoseconds via ImageProxy.imageInfo.timestamp)
 * Depth frames come from Camera2 (nanoseconds via Image.timestamp)
 *
 * Frames are matched within a configurable tolerance (default 33ms = 1 frame @ 30fps)
 */
class DepthFrameSynchronizer(
    private val syncToleranceNs: Long = 33_000_000L,  // 33ms in nanoseconds
    private val maxBufferSize: Int = 5
) {
    companion object {
        private const val TAG = "DepthFrameSynchronizer"
    }

    /**
     * A synchronized RGB + Depth frame pair
     */
    data class SyncedFrame(
        val rgb: Bitmap,
        val rgbTimestamp: Long,
        val depth: DepthCameraManager.DepthFrame,
        val timeDeltaNs: Long,  // Time difference between RGB and Depth
        val syncQuality: SyncQuality
    ) {
        val timeDeltaMs: Float get() = timeDeltaNs / 1_000_000f
    }

    enum class SyncQuality {
        EXCELLENT,  // <10ms delta
        GOOD,       // 10-20ms delta
        ACCEPTABLE, // 20-33ms delta
        POOR        // >33ms but still within tolerance
    }

    // Buffers for unmatched frames
    private data class RgbEntry(val bitmap: Bitmap, val timestamp: Long)
    private val rgbBuffer = ConcurrentHashMap<Long, RgbEntry>()
    private val depthBuffer = ConcurrentHashMap<Long, DepthCameraManager.DepthFrame>()

    // Output flow for synchronized frames
    private val _synchronizedFrames = MutableSharedFlow<SyncedFrame>(
        replay = 0,
        extraBufferCapacity = 2,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val synchronizedFrames: SharedFlow<SyncedFrame> = _synchronizedFrames
    val syncedFrames: SharedFlow<SyncedFrame> = _synchronizedFrames  // Alias

    // Stats
    private var totalRgbFrames = 0L
    private var totalDepthFrames = 0L
    private var totalSyncedFrames = 0L
    private var totalDroppedFrames = 0L

    /**
     * Submit an RGB frame for synchronization
     * @param bitmap The RGB bitmap
     * @param timestampNs Timestamp in nanoseconds (from ImageProxy.imageInfo.timestamp)
     */
    fun submitRgbFrame(bitmap: Bitmap, timestampNs: Long) {
        totalRgbFrames++

        // Try to find matching depth frame
        val matchingDepth = findMatchingDepthFrame(timestampNs)

        if (matchingDepth != null) {
            emitSyncedFrame(bitmap, timestampNs, matchingDepth)
        } else {
            // Buffer for later matching
            rgbBuffer[timestampNs] = RgbEntry(bitmap.copy(bitmap.config, false), timestampNs)
            cleanupOldEntries(timestampNs)
        }
    }

    /**
     * Convenience alias for submitRgbFrame
     */
    fun onRgbFrame(bitmap: Bitmap, timestampNs: Long) = submitRgbFrame(bitmap, timestampNs)

    /**
     * Submit a depth frame for synchronization
     * @param depthFrame The depth frame from DepthCameraManager
     */
    fun submitDepthFrame(depthFrame: DepthCameraManager.DepthFrame) {
        totalDepthFrames++

        // Try to find matching RGB frame
        val matchingRgb = findMatchingRgbFrame(depthFrame.timestamp)

        if (matchingRgb != null) {
            emitSyncedFrame(matchingRgb.bitmap, matchingRgb.timestamp, depthFrame)
            rgbBuffer.remove(matchingRgb.timestamp)
        } else {
            // Buffer for later matching
            depthBuffer[depthFrame.timestamp] = depthFrame
            cleanupOldEntries(depthFrame.timestamp)
        }
    }

    /**
     * Convenience alias for submitDepthFrame
     */
    fun onDepthFrame(depthFrame: DepthCameraManager.DepthFrame) = submitDepthFrame(depthFrame)

    private fun findMatchingDepthFrame(rgbTimestamp: Long): DepthCameraManager.DepthFrame? {
        return depthBuffer.entries
            .filter { abs(it.key - rgbTimestamp) <= syncToleranceNs }
            .minByOrNull { abs(it.key - rgbTimestamp) }
            ?.let { entry ->
                depthBuffer.remove(entry.key)
                entry.value
            }
    }

    private fun findMatchingRgbFrame(depthTimestamp: Long): RgbEntry? {
        return rgbBuffer.entries
            .filter { abs(it.key - depthTimestamp) <= syncToleranceNs }
            .minByOrNull { abs(it.key - depthTimestamp) }
            ?.value
    }

    private fun emitSyncedFrame(
        rgb: Bitmap,
        rgbTimestamp: Long,
        depth: DepthCameraManager.DepthFrame
    ) {
        val timeDelta = abs(rgbTimestamp - depth.timestamp)
        val quality = when {
            timeDelta < 10_000_000 -> SyncQuality.EXCELLENT
            timeDelta < 20_000_000 -> SyncQuality.GOOD
            timeDelta < 33_000_000 -> SyncQuality.ACCEPTABLE
            else -> SyncQuality.POOR
        }

        val syncedFrame = SyncedFrame(
            rgb = rgb,
            rgbTimestamp = rgbTimestamp,
            depth = depth,
            timeDeltaNs = timeDelta,
            syncQuality = quality
        )

        totalSyncedFrames++

        if (totalSyncedFrames % 100 == 0L) {
            Log.d(TAG, "Sync stats: synced=$totalSyncedFrames, " +
                    "rgb=$totalRgbFrames, depth=$totalDepthFrames, " +
                    "dropped=$totalDroppedFrames, " +
                    "rate=${(totalSyncedFrames.toFloat() / totalRgbFrames * 100).toInt()}%")
        }

        _synchronizedFrames.tryEmit(syncedFrame)
    }

    private fun cleanupOldEntries(currentTimestamp: Long) {
        val cutoff = currentTimestamp - (syncToleranceNs * maxBufferSize)

        // Cleanup old RGB entries
        val oldRgbKeys = rgbBuffer.keys.filter { it < cutoff }
        oldRgbKeys.forEach { key ->
            rgbBuffer.remove(key)?.bitmap?.recycle()
            totalDroppedFrames++
        }

        // Cleanup old depth entries
        val oldDepthKeys = depthBuffer.keys.filter { it < cutoff }
        oldDepthKeys.forEach { key ->
            depthBuffer.remove(key)
            totalDroppedFrames++
        }
    }

    /**
     * Get synchronization statistics
     */
    fun getStats(): SyncStats {
        val syncRate = if (totalRgbFrames > 0) {
            totalSyncedFrames.toFloat() / totalRgbFrames
        } else 0f

        return SyncStats(
            totalRgbFrames = totalRgbFrames,
            totalDepthFrames = totalDepthFrames,
            totalSyncedFrames = totalSyncedFrames,
            totalDroppedFrames = totalDroppedFrames,
            syncRate = syncRate,
            rgbBufferSize = rgbBuffer.size,
            depthBufferSize = depthBuffer.size
        )
    }

    data class SyncStats(
        val totalRgbFrames: Long,
        val totalDepthFrames: Long,
        val totalSyncedFrames: Long,
        val totalDroppedFrames: Long,
        val syncRate: Float,
        val rgbBufferSize: Int,
        val depthBufferSize: Int
    )

    /**
     * Reset all buffers and statistics
     */
    fun reset() {
        rgbBuffer.values.forEach { it.bitmap.recycle() }
        rgbBuffer.clear()
        depthBuffer.clear()
        totalRgbFrames = 0
        totalDepthFrames = 0
        totalSyncedFrames = 0
        totalDroppedFrames = 0
    }

    /**
     * Release resources
     */
    fun release() {
        reset()
    }

    /**
     * Convenience alias for release
     */
    fun clear() = release()
}
