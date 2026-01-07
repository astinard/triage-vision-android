package com.triage.vision.camera

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Manages ToF depth sensor capture via Camera2 API.
 *
 * CameraX does not support depth sensors, so we use Camera2 directly
 * for DEPTH16 capture while keeping CameraX for RGB.
 */
class DepthCameraManager(private val context: Context) {

    companion object {
        private const val TAG = "DepthCameraManager"
        private const val LOCK_TIMEOUT_MS = 2500L
    }

    /**
     * Represents a single depth frame from the ToF sensor
     */
    data class DepthFrame(
        val width: Int,
        val height: Int,
        val depthData: ShortArray,
        val timestamp: Long,
        val format: DepthFormat = DepthFormat.DEPTH16_MILLIMETERS
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as DepthFrame
            return width == other.width &&
                   height == other.height &&
                   timestamp == other.timestamp &&
                   depthData.contentEquals(other.depthData)
        }

        override fun hashCode(): Int {
            var result = width
            result = 31 * result + height
            result = 31 * result + depthData.contentHashCode()
            result = 31 * result + timestamp.hashCode()
            return result
        }
    }

    enum class DepthFormat {
        DEPTH16_MILLIMETERS,  // Values are direct millimeter measurements
        DEPTH16_NORMALIZED    // Values are normalized (device-specific)
    }

    data class DepthSensorInfo(
        val cameraId: String,
        val resolution: Size,
        val format: DepthFormat,
        val minDepthMm: Int,
        val maxDepthMm: Int
    )

    // State
    private var cameraManager: CameraManager? = null
    private var depthCameraId: String? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private val cameraOpenCloseLock = Semaphore(1)
    private var isCapturing = false

    // Depth frame output
    private val _latestDepthFrame = MutableStateFlow<DepthFrame?>(null)
    val latestDepthFrame: StateFlow<DepthFrame?> = _latestDepthFrame

    // Stream of all depth frames for synchronization
    private val _depthFrames = MutableSharedFlow<DepthFrame>(
        replay = 0,
        extraBufferCapacity = 2,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val depthFrames: SharedFlow<DepthFrame> = _depthFrames

    private var onDepthFrameCallback: ((DepthFrame) -> Unit)? = null
    private val scope = CoroutineScope(Dispatchers.Default)

    /**
     * Initialize and check for depth sensor availability
     */
    fun initialize(): Boolean {
        cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        depthCameraId = findDepthCamera()

        if (depthCameraId != null) {
            Log.i(TAG, "Depth sensor found: $depthCameraId")
            return true
        } else {
            Log.w(TAG, "No depth sensor available on this device")
            return false
        }
    }

    /**
     * Check if device has a ToF depth sensor
     */
    fun isDepthSupported(): Boolean {
        return depthCameraId != null
    }

    /**
     * Get information about the depth sensor
     */
    fun getDepthSensorInfo(): DepthSensorInfo? {
        val cameraId = depthCameraId ?: return null
        val manager = cameraManager ?: return null

        try {
            val characteristics = manager.getCameraCharacteristics(cameraId)
            val configs = characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
            ) ?: return null

            val depthSizes = configs.getOutputSizes(ImageFormat.DEPTH16)
            if (depthSizes.isNullOrEmpty()) return null

            // Use largest available resolution
            val resolution = depthSizes.maxByOrNull { it.width * it.height } ?: return null

            return DepthSensorInfo(
                cameraId = cameraId,
                resolution = resolution,
                format = DepthFormat.DEPTH16_MILLIMETERS,
                minDepthMm = 100,   // Typical ToF min range
                maxDepthMm = 5000   // Typical ToF max range (5m)
            )
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to get depth sensor info", e)
            return null
        }
    }

    /**
     * Start capturing depth frames
     */
    suspend fun startDepthCapture(
        onDepthFrame: (DepthFrame) -> Unit
    ): Result<Unit> = suspendCancellableCoroutine { continuation ->
        if (depthCameraId == null) {
            continuation.resume(Result.failure(IllegalStateException("No depth sensor available")))
            return@suspendCancellableCoroutine
        }

        if (isCapturing) {
            Log.w(TAG, "Depth capture already running")
            continuation.resume(Result.success(Unit))
            return@suspendCancellableCoroutine
        }

        onDepthFrameCallback = onDepthFrame

        try {
            startBackgroundThread()
            openCamera { success ->
                if (success) {
                    isCapturing = true
                    continuation.resume(Result.success(Unit))
                } else {
                    continuation.resume(Result.failure(
                        IllegalStateException("Failed to open depth camera")
                    ))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start depth capture", e)
            continuation.resume(Result.failure(e))
        }
    }

    /**
     * Stop depth capture and release resources
     */
    fun stopDepthCapture() {
        Log.i(TAG, "Stopping depth capture")
        isCapturing = false
        closeCamera()
        stopBackgroundThread()
        onDepthFrameCallback = null
    }

    /**
     * Convenience method to start depth capture without callback
     * Frames will be available via depthFrames SharedFlow
     */
    fun start() {
        if (depthCameraId == null) {
            Log.w(TAG, "Cannot start: no depth sensor available")
            return
        }
        if (isCapturing) {
            Log.w(TAG, "Depth capture already running")
            return
        }

        scope.launch {
            startDepthCapture { /* frames go to SharedFlow */ }
        }
    }

    /**
     * Convenience method to stop depth capture
     */
    fun stop() {
        stopDepthCapture()
    }

    /**
     * Find camera with DEPTH16 output capability
     */
    private fun findDepthCamera(): String? {
        val manager = cameraManager ?: return null

        try {
            for (cameraId in manager.cameraIdList) {
                val characteristics = manager.getCameraCharacteristics(cameraId)

                // Check for DEPTH_OUTPUT capability
                val capabilities = characteristics.get(
                    CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES
                )

                if (capabilities?.contains(
                        CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT
                    ) == true
                ) {
                    // Verify DEPTH16 format is available
                    val configs = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
                    )

                    val depthSizes = configs?.getOutputSizes(ImageFormat.DEPTH16)
                    if (!depthSizes.isNullOrEmpty()) {
                        Log.i(TAG, "Found ToF camera: $cameraId with sizes: ${depthSizes.toList()}")
                        return cameraId
                    }
                }
            }
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Camera access error while finding depth sensor", e)
        }

        return null
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("DepthCameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, "Background thread interrupted", e)
        }
    }

    private fun openCamera(onComplete: (Boolean) -> Unit) {
        val cameraId = depthCameraId ?: run {
            onComplete(false)
            return
        }

        val manager = cameraManager ?: run {
            onComplete(false)
            return
        }

        try {
            if (!cameraOpenCloseLock.tryAcquire(LOCK_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                throw RuntimeException("Timeout waiting to lock camera opening")
            }

            val sensorInfo = getDepthSensorInfo() ?: run {
                cameraOpenCloseLock.release()
                onComplete(false)
                return
            }

            // Create ImageReader for DEPTH16 format
            imageReader = ImageReader.newInstance(
                sensorInfo.resolution.width,
                sensorInfo.resolution.height,
                ImageFormat.DEPTH16,
                2 // Double buffer
            ).apply {
                setOnImageAvailableListener({ reader ->
                    val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                    processDepthImage(image)
                    image.close()
                }, backgroundHandler)
            }

            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraOpenCloseLock.release()
                    cameraDevice = camera
                    createCaptureSession(onComplete)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    cameraOpenCloseLock.release()
                    camera.close()
                    cameraDevice = null
                    onComplete(false)
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    cameraOpenCloseLock.release()
                    camera.close()
                    cameraDevice = null
                    Log.e(TAG, "Camera device error: $error")
                    onComplete(false)
                }
            }, backgroundHandler)

        } catch (e: CameraAccessException) {
            Log.e(TAG, "Camera access exception", e)
            cameraOpenCloseLock.release()
            onComplete(false)
        } catch (e: SecurityException) {
            Log.e(TAG, "Camera permission denied", e)
            cameraOpenCloseLock.release()
            onComplete(false)
        }
    }

    private fun createCaptureSession(onComplete: (Boolean) -> Unit) {
        val camera = cameraDevice ?: run {
            onComplete(false)
            return
        }

        val reader = imageReader ?: run {
            onComplete(false)
            return
        }

        try {
            val surfaces = listOf(reader.surface)

            camera.createCaptureSession(
                surfaces,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        startRepeatingRequest()
                        onComplete(true)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Capture session configuration failed")
                        onComplete(false)
                    }
                },
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to create capture session", e)
            onComplete(false)
        }
    }

    private fun startRepeatingRequest() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return
        val reader = imageReader ?: return

        try {
            val captureRequest = camera.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            ).apply {
                addTarget(reader.surface)
            }.build()

            session.setRepeatingRequest(captureRequest, null, backgroundHandler)
            Log.i(TAG, "Depth capture started")

        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to start repeating request", e)
        }
    }

    private fun processDepthImage(image: Image) {
        if (!isCapturing) return

        val plane = image.planes[0]
        val buffer = plane.buffer
        val width = image.width
        val height = image.height
        val timestamp = image.timestamp

        // DEPTH16 is 16-bit per pixel (2 bytes)
        val depthData = ShortArray(width * height)
        buffer.rewind()
        buffer.asShortBuffer().get(depthData)

        val depthFrame = DepthFrame(
            width = width,
            height = height,
            depthData = depthData,
            timestamp = timestamp,
            format = DepthFormat.DEPTH16_MILLIMETERS
        )

        // Update state, emit to flow, and callback
        _latestDepthFrame.value = depthFrame
        _depthFrames.tryEmit(depthFrame)
        onDepthFrameCallback?.invoke(depthFrame)
    }

    private fun closeCamera() {
        try {
            cameraOpenCloseLock.acquire()

            captureSession?.close()
            captureSession = null

            cameraDevice?.close()
            cameraDevice = null

            imageReader?.close()
            imageReader = null

        } catch (e: InterruptedException) {
            throw RuntimeException("Interrupted while closing camera", e)
        } finally {
            cameraOpenCloseLock.release()
        }
    }

    /**
     * Release all resources - call when done with depth capture
     */
    fun release() {
        stopDepthCapture()
        cameraManager = null
        depthCameraId = null
    }
}
