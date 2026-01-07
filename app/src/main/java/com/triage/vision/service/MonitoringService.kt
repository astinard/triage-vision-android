package com.triage.vision.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.triage.vision.R
import com.triage.vision.TriageVisionApp
import com.triage.vision.ui.MainActivity

/**
 * Foreground service for continuous patient monitoring.
 *
 * Required for:
 * - Camera access in background
 * - Persistent notifications
 * - Reliable alert delivery
 */
class MonitoringService : Service() {

    companion object {
        private const val TAG = "MonitoringService"
        private const val NOTIFICATION_ID = 1001

        const val ACTION_START = "com.triage.vision.START_MONITORING"
        const val ACTION_STOP = "com.triage.vision.STOP_MONITORING"
    }

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "MonitoringService created")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startMonitoring()
            ACTION_STOP -> stopMonitoring()
        }
        return START_STICKY
    }

    private fun startMonitoring() {
        Log.i(TAG, "Starting foreground monitoring")

        val notification = createNotification()
        startForeground(NOTIFICATION_ID, notification)

        // Initialize pipelines
        val app = TriageVisionApp.instance
        if (!app.isNativeInitialized) {
            Log.w(TAG, "Native libraries not initialized")
        }
    }

    private fun stopMonitoring() {
        Log.i(TAG, "Stopping monitoring service")
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun createNotification(): Notification {
        // Intent to open app
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val openPendingIntent = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        // Intent to stop monitoring
        val stopIntent = Intent(this, MonitoringService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, TriageVisionApp.CHANNEL_MONITORING)
            .setContentTitle(getString(R.string.notification_monitoring_title))
            .setContentText(getString(R.string.notification_monitoring_text))
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .setContentIntent(openPendingIntent)
            .addAction(
                android.R.drawable.ic_media_pause,
                "Stop",
                stopPendingIntent
            )
            .build()
    }

    /**
     * Send alert notification
     */
    fun sendAlertNotification(title: String, message: String) {
        val notification = NotificationCompat.Builder(this, TriageVisionApp.CHANNEL_ALERTS)
            .setContentTitle(title)
            .setContentText(message)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setAutoCancel(true)
            .build()

        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as android.app.NotificationManager
        notificationManager.notify(NOTIFICATION_ID + 1, notification)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "MonitoringService destroyed")
    }
}
